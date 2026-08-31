add_dataflow_scheduler_library(DataflowSchedulerKTDFTransforms
    StageCoarsening/BufferExpansion.cpp
    StageCoarsening/Materializer.cpp
    StageCoarsening/ScopeCorrection.cpp
    BroadcastPromotion.cpp
    MapReductionPartials.cpp
    ReductionOpacification.cpp
    ReductionDimChunking.cpp
    ReductionUtils.cpp
    ReductionLoopExposure.cpp
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
)
