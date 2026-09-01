add_dataflow_scheduler_library(DataflowSchedulerKTDFLoweringDialect
    KTDFLoweringDialect.cpp
    KTDFLoweringOps.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDFLowering
    ${DataflowScheduler_BINARY_DIR}/include/dataflow-scheduler/Dialect/KTDFLowering

  GROUP DIALECT
  EXPORT_NAME KTDFLoweringDialect

  DEPENDS
    DataflowSchedulerKTDFLoweringIncGen

  LINK_LIBS PUBLIC
    DataflowSchedulerDialects::KTDFDialect

  LINK_MLIR_LIBS
    MLIRIR
    MLIRControlFlowInterfaces
)
