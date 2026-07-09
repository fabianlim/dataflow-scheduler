add_dataflow_scheduler_library(DataflowSchedulerKTIRToScheduleIR
    ComputeGroupExtraction.cpp
    ConstructThreeStagePipeline.cpp
    KTIRLegalityCheck.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/frontend/KTIRToScheduleIR

  GROUP CONVERSION
  EXPORT_NAME KTIRToScheduleIR

  DEPENDS
    DataflowSchedulerKTIRToScheduleIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFTransforms
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils
    DataflowScheduler::Utils
)

mlir_target_link_libraries(DataflowSchedulerKTIRToScheduleIR PUBLIC
  MLIRIR
  MLIRArithDialect
  MLIRBufferizationDialect
  MLIRLinalgDialect
  MLIRMathDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransformUtils
)
