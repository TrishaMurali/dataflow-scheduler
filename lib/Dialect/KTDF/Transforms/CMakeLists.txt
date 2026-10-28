add_dataflow_scheduler_library(DataflowSchedulerKTDFTransforms
    LoopIVAffineMulStrengthReduction.cpp
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/Materializer.cpp
    StageCoarsening/ScopeCorrection.cpp
    BroadcastPromotion.cpp
    MappableExtensions.cpp
    MapReductionPartials.cpp
    ReductionDimChunking.cpp
    ReductionUtils.cpp
    ReductionLoopExposure.cpp
    SimplifyPipeline.cpp
    SplitReductionInnerOuterDim.cpp
    StageCoarsening.cpp
    SubsumeLinearizeIndex.cpp
    TileNormalized.cpp
    TileSizeSelection.cpp

  ADDITIONAL_HEADER_DIRS
    ${DataflowScheduler_SOURCE_DIR}/include/dataflow-scheduler/Dialect/KTDF/Transforms

  GROUP DIALECT
  EXPORT_NAME KTDFTransforms

  DEPENDS
    DataflowSchedulerKTDFTransformsPassesIncGen

  LINK_LIBS PUBLIC
    DataflowScheduler::ArchViews
    DataflowScheduler::KTDFAnalysis
    DataflowSchedulerDialects::KTDFDialect
    DataflowSchedulerDialects::KTDFTransforms
    DataflowSchedulerDialects::KTDFArchAnalysis
    DataflowScheduler::KTDFUtils
    DataflowScheduler::TransformUtils

  LINK_MLIR_LIBS
    MLIRAffineDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRLinalgDialect
    MLIRMemRefDialect
    MLIRPass
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRDialectUtils
)
