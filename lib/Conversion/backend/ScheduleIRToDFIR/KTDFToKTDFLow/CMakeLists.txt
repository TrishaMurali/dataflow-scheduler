add_dataflow_scheduler_library(DataflowSchedulerKTDFToKTDFLow
    ComponentClassifier.cpp
    KTDFToKTDFLow.cpp
    PipelineExecutionTransform.cpp
    ScratchpadConflicts.cpp
    SignalInsertion.cpp
    UniformInfra.cpp
    UnitMaterializer.cpp
    UnitSSACollection.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Conversion/backend/ScheduleIRToDFIR/KTDFToKTDFLow

  GROUP CONVERSION
  EXPORT_NAME KTDFToKTDFLow

  DEPENDS
    DataflowSchedulerScheduleIRToDFIRPassesIncGen

  LINK_LIBS PUBLIC
    KTIR::KtdpDialect
    DataflowSchedulerDialects::AgenDialect
    DataflowScheduler::Analysis
    DataflowScheduler::ConversionUtils
    DataflowSchedulerDialects::DataflowDialect
    DataflowSchedulerDialects::KTDFDialect
    DataflowScheduler::KTDFAnalysis
    DataflowScheduler::KTDFUtils
    DataflowScheduler::KTDFLoweringDialect
    DataflowScheduler::Transforms
    DataflowScheduler::TransformUtils
    DataflowSchedulerDialects::UniformDialect
    DataflowScheduler::Utils

  LINK_MLIR_LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRPass
)

mlir_target_link_libraries(DataflowSchedulerKTDFToKTDFLow PUBLIC
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
)
