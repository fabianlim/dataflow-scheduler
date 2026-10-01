add_subdirectory(ArchViews)

add_dataflow_scheduler_library(DataflowSchedulerAnalysis
    CrossCoreChannels.cpp
    InvarianceCheck.cpp
    MemoryTracker.cpp
    MemoryTrackerAnalysis.cpp
    OperationTree.cpp
    PipelineTree.cpp
    Utils.cpp
    WriteSetScan.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Analysis

  EXPORT_NAME Analysis

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFArchDialect
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::ArchViews
    KTIR::KtdpDialect

  LINK_MLIR_LIBS
    MLIRAffineAnalysis
    MLIRAnalysis
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRSCFDialect
)
