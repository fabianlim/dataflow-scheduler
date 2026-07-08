//===----------------------------------------------------------------------===//
//
// Part of the Dataflow Scheduler project.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
/// Operation lowerings for KTDFLowToDFIR pass
///
//
//===----------------------------------------------------------------------===//

#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/OperationLowerings.h"

#include "dataflow-scheduler/Analysis/ArchViews/ResourceKinds.h"
#include "dataflow-scheduler/Conversion/Utils/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/BufferPhaseLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/DataTransferLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/LinalgLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/ParallelLowering.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFLowToDFIR/Utils.h"
#include "dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow/UniformInfra.h"
#include "dataflow-scheduler/Dialect/Agen/Agen.h"
#include "dataflow-scheduler/Dialect/Agen/Utils.h"
#include "dataflow-scheduler/Dialect/Dataflow/Dataflow.h"
#include "dataflow-scheduler/Dialect/Dataflow/Utils.h"
#include "dataflow-scheduler/Dialect/KTDF/KTDF.h"
#include "dataflow-scheduler/Dialect/KTDFLowering/KTDFLowering.h"
#include "dataflow-scheduler/Dialect/Uniform/Uniform.h"
#include "dataflow-scheduler/Utils/SchedulerExtContext.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/IntegerSet.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "ktdflowering-to-dfir"

using namespace scheduler;

namespace {

// Defined below; forward-declared so LowerWriteToFifoPattern can vector_load the
// post-loop reduction accumulator from its lrfreg view.
static mlir::Value emitVectorLoad(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value view);

/// Pattern to lower ktdf.read_from_fifo operations
struct LowerReadFromFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::ReadFromFifoOp> {
  LowerReadFromFifoPattern(mlir::MLIRContext* context,
                           const SchedulerExtContext& scheduler_ctx,
                           arch_view::ResourceKinds& resource_kinds,
                           const ResourceToUnits& components)
      : OpRewritePattern(context),
        scheduler_ctx_(scheduler_ctx),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::ReadFromFifoOp read_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(read_op.getFifoSlot().getType());

    // Get the result tensor type
    auto result_type = llvm::cast<mlir::RankedTensorType>(read_op.getType());

    // Convert tensor to flattened vector type
    auto vector_type = getFlattenedVectorType(result_type, resource_kinds_);
    if (!vector_type) {
      return mlir::failure();
    }

    // Extract source component type from FIFO slot (read from source)
    // The source is the producer side, so we use the src attribute
    mlir::Attribute src_attr = fifo_slot_type.getSrc();
    std::optional<scheduler::ResourceType> component_type_opt;

    // The resource is the source's type name as a StringAttr.
    if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(src_attr)) {
      component_type_opt = mlir::StringAttr::get(str_attr.getContext(),
                                                 str_attr.getValue().upper());
    }
    if (!component_type_opt.has_value()) {
      read_op.emitError("unsupported read_from_fifo transfer");
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        read_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      read_op.emitError("read_from_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Get target component units
    auto it = components_.find(*component_type_opt);
    if (it == components_.end()) {
      read_op.emitError("no units found for source component type");
      return mlir::failure();
    }
    const auto& target_units = it->second;

    // Create query_map to map from current program_unit operands to target
    // units
    mlir::Value queried_unit = createQueryMapForComponent(
        rewriter, program_unit, target_units, read_op.getLoc());

    // Create dataflow.receive operation
    auto receive_op = mlir::dataflow::ReceiveOp::create(
        rewriter, read_op.getLoc(), vector_type, queried_unit,
        /*dbgName=*/nullptr);

    // Replace the read_from_fifo with the receive operation
    rewriter.replaceOp(read_op, receive_op.getData());

    return mlir::success();
  }

 private:
  const SchedulerExtContext& scheduler_ctx_;
  arch_view::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

struct LowerWriteToFifoPattern
    : public mlir::OpRewritePattern<mlir::ktdf::WriteToFifoOp> {
  LowerWriteToFifoPattern(mlir::MLIRContext* context,
                          const SchedulerExtContext& scheduler_ctx,
                          arch_view::ResourceKinds& resource_kinds,
                          const ResourceToUnits& components)
      : OpRewritePattern(context),
        scheduler_ctx_(scheduler_ctx),
        resource_kinds_(resource_kinds),
        components_(components) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::WriteToFifoOp write_op,
      mlir::PatternRewriter& rewriter) const override {
    // Get the FIFO slot type
    auto fifo_slot_type =
        llvm::cast<mlir::ktdf::FifoSlotType>(write_op.getFifoSlot().getType());

    // Resolve the data to send. Post-loop reduction accumulator store: the data
    // is insert_slice(to_tensor(lrfreg view)) — a tensor that no other pattern
    // vectorizes. Load the accumulator from its view as a vector and bypass the
    // insert_slice (the FIFO/send flattens to the lane count anyway). The lrfreg
    // view is recognized by its backing get_local_unit {name="lrfreg"}.
    mlir::Value data = write_op.getData();
    if (auto ins = data.getDefiningOp<mlir::tensor::InsertSliceOp>()) {
      if (auto tt =
              ins.getSource().getDefiningOp<mlir::bufferization::ToTensorOp>()) {
        if (auto view = tt.getBuffer()
                            .getDefiningOp<mlir::dataflow::GetLogicalMemoryViewOp>()) {
          if (auto lu = view.getFromUnit()
                            .getDefiningOp<mlir::dataflow::GetLocalUnitOp>()) {
            if (lu.getName() == "lrfreg") {
              data = emitVectorLoad(rewriter, write_op.getLoc(), tt.getBuffer());
            }
          }
        }
      }
    }

    // Convert data type (tensor or vector) to flattened vector type
    auto vector_type = getFlattenedVectorType(data.getType(), resource_kinds_);
    if (!vector_type) {
      return mlir::failure();
    }

    // Extract destination component type from FIFO slot (write to destination)
    // The destination is the consumer side, so we use the dest attribute
    mlir::Attribute dest_attr = fifo_slot_type.getDest();
    std::optional<scheduler::ResourceType> component_type_opt;

    // The resource is the destination's type name as a StringAttr.
    if (auto str_attr = mlir::dyn_cast<mlir::StringAttr>(dest_attr)) {
      component_type_opt = mlir::StringAttr::get(str_attr.getContext(),
                                                 str_attr.getValue().upper());
    }
    if (!component_type_opt.has_value()) {
      write_op.emitError("unsupported write_to_fifo transfer");
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        write_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      write_op.emitError("write_to_fifo must be inside a program_unit");
      return mlir::failure();
    }

    // Get target component units
    auto it = components_.find(*component_type_opt);
    if (it == components_.end()) {
      write_op.emitError("no units found for destination component type");
      return mlir::failure();
    }
    const auto& target_units = it->second;

    // Create query_map to map from current program_unit operands to target
    // units
    mlir::Value queried_unit = createQueryMapForComponent(
        rewriter, program_unit, target_units, write_op.getLoc());

    // Create dataflow.send operation
    mlir::dataflow::SendOp::create(rewriter, write_op.getLoc(), queried_unit,
                                   data, /*dir=*/nullptr,
                                   /*dbgName=*/nullptr);

    // Erase the write_to_fifo operation
    rewriter.eraseOp(write_op);

    return mlir::success();
  }

 private:
  const SchedulerExtContext& scheduler_ctx_;
  arch_view::ResourceKinds& resource_kinds_;
  const ResourceToUnits& components_;
};

/// Helper to find the current unit's query_map from signal operands
/// Returns the query_map whose first mapping result matches a program_unit
/// operand
static llvm::FailureOr<mlir::Value> findCurrentUnitQueryMap(
    llvm::ArrayRef<mlir::Value> signal_units,
    llvm::ArrayRef<mlir::Value> program_unit_operands, mlir::Operation* op) {
  for (auto signal_unit : signal_units) {
    // Get the query_map operation
    auto query_op = signal_unit.getDefiningOp<mlir::uniform::QueryMapOp>();
    if (!query_op) {
      op->emitError("signal operand must be a uniform.query_map result");
      return mlir::failure();
    }

    // Get the def_immutable_mapping
    auto def_map_op =
        query_op.getMap().getDefiningOp<mlir::uniform::DefImmutableMappingOp>();
    if (!def_map_op) {
      op->emitError("query_map must reference a def_immutable_mapping");
      return mlir::failure();
    }

    // Get the first value from the mapping (first result unit)
    auto values = def_map_op.getValues();
    if (values.empty()) {
      op->emitError("def_immutable_mapping must have at least one value");
      return mlir::failure();
    }
    mlir::Value first_unit = values[0];

    // Check if this first unit is in the current program_unit operands
    for (auto pu_unit : program_unit_operands) {
      if (first_unit == pu_unit) {
        return signal_unit;
      }
    }
  }

  op->emitError(
      "signal must include at least one query_map whose first mapping result "
      "is in the current program_unit");
  return mlir::failure();
}
/// Pattern to lower ktdf.tiling.derive_size operations using conditional
/// branching
struct LowerGetTileSizePattern
    : public mlir::OpRewritePattern<mlir::ktdf::TilingDeriveSizeOp> {
  LowerGetTileSizePattern(mlir::MLIRContext* context,
                          const SchedulerExtContext& /*scheduler_ctx*/,
                          const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf::TilingDeriveSizeOp derive_size_op,
      mlir::PatternRewriter& rewriter) const override {
    auto loc = derive_size_op.getLoc();
    auto ivs = derive_size_op.getIvs();
    auto tile_sizes = derive_size_op.getTileSizes();
    auto total_size = derive_size_op.getTotalSize();

    // We only handle the single-level case (one [iv : tile_size] pair)
    if (ivs.size() != 1) {
      return mlir::failure();
    }

    auto iv = ivs[0];
    auto tile_size = tile_sizes[0];

    // Find the enclosing scf.for loop that uses this IV
    auto iv_block_arg = llvm::dyn_cast<mlir::BlockArgument>(iv);
    if (!iv_block_arg) {
      derive_size_op.emitError("IV must be a block argument");
      return mlir::failure();
    }

    auto for_op = iv_block_arg.getOwner()->getParentOp();
    auto scf_for = mlir::dyn_cast<mlir::scf::ForOp>(for_op);

    if (!scf_for) {
      derive_size_op.emitError("IV must be from an scf.for loop");
      return mlir::failure();
    }
    // Extract constant values - both must be constants
    auto tile_size_const =
        tile_size.getDefiningOp<mlir::arith::ConstantIndexOp>();
    auto total_size_const =
        total_size.getDefiningOp<mlir::arith::ConstantIndexOp>();

    if (!tile_size_const || !total_size_const) {
      derive_size_op.emitError("tile_size and total_size must be constants");
      return mlir::failure();
    }

    int64_t tile_size_val = tile_size_const.value();
    int64_t total_size_val = total_size_const.value();

    // Epilogue size is the remainder; zero means total_size divides tile_size
    // evenly and every iteration — including the last — uses the steady-state
    // tile size.
    int64_t epilogue_size_val = total_size_val % tile_size_val;

    mlir::Value result;
    if (epilogue_size_val == 0) {
      // No epilogue: every tile has the steady-state size; no conditional
      // needed.
      result = tile_size;
    } else {
      // Epilogue exists: last iteration gets epilogue_size, others get
      // tile_size.
      auto upper_bound = scf_for.getUpperBound();
      auto c1 = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      auto upper_bound_minus_1 =
          mlir::arith::SubIOp::create(rewriter, loc, upper_bound, c1);
      auto is_not_last = mlir::arith::CmpIOp::create(
          rewriter, loc, mlir::arith::CmpIPredicate::slt, iv,
          upper_bound_minus_1);
      auto epilogue_size = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, epilogue_size_val);
      result = mlir::arith::SelectOp::create(rewriter, loc, is_not_last,
                                             tile_size, epilogue_size);
    }

    rewriter.replaceOp(derive_size_op, result);

    return mlir::success();
  }
};

/// Pattern to lower ktdf_lowering.signal operations
struct LowerSignalPattern
    : public mlir::OpRewritePattern<mlir::ktdf_lowering::SignalOp> {
  LowerSignalPattern(mlir::MLIRContext* context,
                     const SchedulerExtContext& /*scheduler_ctx*/,
                     const ResourceToUnits& /*components*/)
      : OpRewritePattern(context) {}

  mlir::LogicalResult matchAndRewrite(
      mlir::ktdf_lowering::SignalOp signal_op,
      mlir::PatternRewriter& rewriter) const override {
    // Signal operations must have at least 2 units
    if (signal_op.getNumUnits() < 2) {
      signal_op.emitError(
          "signal operation must have at least 2 unit operands");
      return mlir::failure();
    }

    // Find the enclosing program_unit
    auto program_unit =
        signal_op->getParentOfType<mlir::dataflow::ProgramUnitOp>();
    if (!program_unit) {
      signal_op.emitError("signal must be inside a program_unit");
      return mlir::failure();
    }

    auto signal_units = signal_op.getUnits();
    auto program_unit_operands = program_unit.getUnits();

    // Convert to vectors for the helper function
    llvm::SmallVector<mlir::Value, 8> signal_units_vec(signal_units.begin(),
                                                       signal_units.end());
    llvm::SmallVector<mlir::Value, 8> program_unit_vec(
        program_unit_operands.begin(), program_unit_operands.end());

    // Find which signal operand corresponds to the current program_unit
    auto curr_unit_query_map_result = findCurrentUnitQueryMap(
        signal_units_vec, program_unit_vec, signal_op.getOperation());
    if (mlir::failed(curr_unit_query_map_result)) {
      return mlir::failure();
    }
    mlir::Value curr_unit_query_map = *curr_unit_query_map_result;

    auto curr_resource_type =
        scheduler::getUnitTypeFromQueryMap(curr_unit_query_map);
    assert(!curr_resource_type.empty() && "getUnitTypeFromQueryMap failed");

    // Build query maps for all other units (not the current unit)
    // Only include units with different resource types than the current unit
    llvm::SmallVector<mlir::Value, 8> other_query_maps;
    for (auto signal_unit : signal_units) {
      if (signal_unit != curr_unit_query_map) {
        auto other_resource_type =
            scheduler::getUnitTypeFromQueryMap(signal_unit);
        assert(!other_resource_type.empty() &&
               "getUnitTypeFromQueryMap failed");

        // Only sync with units of different types. This avoids syncing between
        // corelets 0 and 1 of the same unit type for example.
        if (other_resource_type != curr_resource_type) {
          auto other_query_map_result = UniformInfra::buildSignalQueryMap(
              signal_unit, program_unit, rewriter, signal_op.getLoc());
          if (mlir::failed(other_query_map_result)) {
            signal_op.emitError("failed to build query map for signal operand");
            return mlir::failure();
          }
          other_query_maps.push_back(*other_query_map_result);
        }
      }
    }

    // Create all sync_send operations (from current unit to other units)
    bool wait_immediately_for_async_transfer = 1;
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncSendOp::create(
          rewriter, signal_op.getLoc(), other_query_map, /*dbgName=*/nullptr,
          rewriter.getBoolAttr(wait_immediately_for_async_transfer));
    }

    // Create all sync_recv operations (from other units to current unit)
    for (mlir::Value other_query_map : other_query_maps) {
      mlir::dataflow::SyncRecvOp::create(rewriter, signal_op.getLoc(),
                                         other_query_map, /*dbgName=*/nullptr);
    }

    // Erase the signal operation
    rewriter.eraseOp(signal_op);

    return mlir::success();
  }
};

/// Build IntegerSet for a 1-D vector-access constraint: d0 in [0, 63].
static mlir::IntegerSet buildLoadStoreSet1D(mlir::MLIRContext* ctx) {
  auto d0 = mlir::getAffineDimExpr(0, ctx);
  // d0 >= 0  &&  -d0 + 63 >= 0
  return mlir::IntegerSet::get(1, 0, {d0, -d0 + 63}, {/*eq=*/false, /*eq=*/false});
}

/// Emit agen.vector_load at `view` with index 0.
/// Returns the loaded vector value.
static mlir::Value emitVectorLoad(mlir::OpBuilder& builder, mlir::Location loc,
                                  mlir::Value view) {
  auto* ctx = builder.getContext();
  auto memref_type = mlir::cast<mlir::MemRefType>(view.getType());
  auto shape = memref_type.getShape();
  assert(shape.size() == 1 && "expected 1-D lrfreg view");
  int64_t num_elems = shape[0];
  auto elem_type = memref_type.getElementType();
  auto vector_type = mlir::VectorType::get({num_elems}, elem_type);
  auto identity1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
  auto load_set = buildLoadStoreSet1D(ctx);
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  return mlir::agen::VectorLoadOp::create(builder, loc, vector_type, view,
                                          /*dbgName=*/nullptr, identity1d,
                                          mlir::ValueRange{c0}, load_set,
                                          identity1d)
      .getResult();
}

/// Emit agen.vector_store of `vec` to `view` at index 0.
static void emitVectorStore(mlir::OpBuilder& builder, mlir::Location loc,
                             mlir::Value vec, mlir::Value view) {
  auto* ctx = builder.getContext();
  auto identity1d = mlir::AffineMap::getMultiDimIdentityMap(1, ctx);
  auto store_set = buildLoadStoreSet1D(ctx);
  mlir::Value c0 = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
  mlir::agen::VectorStoreOp::create(builder, loc, vec, view,
                                    /*dbgName=*/nullptr, identity1d,
                                    mlir::ValueRange{c0}, store_set, identity1d);
}

/// Lower bufferization.materialize_in_destination(%const_tensor, %view)
/// where source is a dense constant (the zero-seed) to agen.vector_store.
/// Only constant-tensor sources are handled here; non-constant sources
/// (the per-iteration accumulator write-back) are handled by LowerLinalgGenericPattern.
static mlir::LogicalResult lowerLrfregBufferizationOpsPhaseA(
    mlir::func::FuncOp func) {
  mlir::OpBuilder builder(func.getContext());

  llvm::SmallVector<mlir::bufferization::MaterializeInDestinationOp>
      materialize_ops;

  func.walk([&](mlir::bufferization::MaterializeInDestinationOp mid) {
    if (mlir::isa<mlir::MemRefType>(mid.getDest().getType()))
      materialize_ops.push_back(mid);
  });

  // Lower materialize_in_destination -> vector_store only when the
  // source is a dense constant (the zero-seed for the lrfreg accumulator).
  for (auto mid : materialize_ops) {
    mlir::Value src = mid.getSource();
    mlir::Value view = mid.getDest();
    auto memref_type = mlir::dyn_cast<mlir::MemRefType>(view.getType());
    if (!memref_type || memref_type.getRank() != 1) continue;

    // Only lower constant-tensor sources here (the zero seed).
    if (!mlir::isa<mlir::TensorType>(src.getType())) continue;
    auto const_op = src.getDefiningOp<mlir::arith::ConstantOp>();
    if (!const_op) continue;

    int64_t num_elems = memref_type.getShape()[0];
    auto elem_type = memref_type.getElementType();
    auto vector_type = mlir::VectorType::get({num_elems}, elem_type);
    auto dense_attr =
        mlir::cast<mlir::DenseElementsAttr>(const_op.getValue());
    auto vec_attr = dense_attr.reshape(vector_type);

    builder.setInsertionPoint(mid);
    mlir::Value vec =
        mlir::arith::ConstantOp::create(builder, mid.getLoc(), vec_attr)
            .getResult();
    emitVectorStore(builder, mid.getLoc(), vec, view);
    mid.erase();
  }

  return mlir::success();
}

}  // namespace

mlir::LogicalResult scheduler::runOperationLowerings(
    mlir::func::FuncOp func,
    const scheduler::SchedulerExtContext& scheduler_ctx,
    const ResourceToUnits& components,
    arch_view::ResourceKinds& resource_kinds) {
  // Lower linalg.generic compute operations and FIFO operations
  mlir::RewritePatternSet patterns(func.getContext());
  populateLinalgLoweringPatterns(patterns, scheduler_ctx, resource_kinds);
  patterns.add<LowerReadFromFifoPattern>(func.getContext(), scheduler_ctx,
                                         resource_kinds, components);
  patterns.add<LowerWriteToFifoPattern>(func.getContext(), scheduler_ctx,
                                        resource_kinds, components);
  populateDataTransferLoweringPatterns(patterns, scheduler_ctx, components);
  patterns.add<LowerSignalPattern>(func.getContext(), scheduler_ctx,
                                   components);
  patterns.add<LowerGetTileSizePattern>(func.getContext(), scheduler_ctx,
                                        components);
  // Disable opportunistic folding in this lowering pass: the greedy driver's
  // fold path hits a use-after-free on the transient reduction IR (bufferization
  // + partially-lowered linalg) present while patterns are being applied.
  // Folding is not needed to apply these lowering patterns, and the final
  // canonicalizeFunc (a separate step) still folds the fully-lowered IR.
  mlir::GreedyRewriteConfig lowering_cfg;
  lowering_cfg.enableFolding(false);
  if (mlir::failed(
          mlir::applyPatternsGreedily(func, std::move(patterns), lowering_cfg))) {
    return mlir::failure();
  }

  // Clone loops for buffer phase tracking and replace operations
  if (mlir::failed(lowerDoubleBuffering(func, components, resource_kinds))) {
    return mlir::failure();
  }

  // Lower ktdf.parallel operations after all other lowerings are complete
  if (mlir::failed(lowerParallelOps(func))) {
    return mlir::failure();
  }

  return mlir::success();
}

// Made with Bob
