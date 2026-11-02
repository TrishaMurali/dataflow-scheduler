// A device where data for the compute unit can take two routes from L1:
//  - L1 -> L1LU -> SFU directly, or
//  - L1 -> L1LU -> L2SU -> L2 -> L2LU -> SFU, where L1LU streams straight
//    into L2SU without a memory in between.
// Results go SFU -> SFUSU -> L1 -> L1SU -> DDR, and DDR -> DDRLU -> L1 feeds
// L1.

#DDR = { kind = "DDR", size = 1073741824 }
#L1 = { kind = "L1", size = 1048576 }
#L2 = { kind = "L2", size = 8192 }

#DDRLU = { kind = "DDRLU", ktdf_arch.features = { ktdf_arch.feature.load = {} } }
#L1SU = { kind = "L1SU", ktdf_arch.features = { ktdf_arch.feature.store = {} } }
#L1LU = { kind = "L1LU", ktdf_arch.features = { ktdf_arch.feature.load = {} } }
#SFUSU = { kind = "SFUSU", ktdf_arch.features = { ktdf_arch.feature.store = {} } }
#L2SU = { kind = "L2SU", ktdf_arch.features = { ktdf_arch.feature.store = {} } }
#L2LU = { kind = "L2LU", ktdf_arch.features = { ktdf_arch.feature.load = {} } }
#SFU = { kind = "SFU", ktdf_arch.features = { ktdf_arch.feature.compute } }

#PATH = {}

ktdf_arch.device @chained_units_device {
  %ddr = memory #DDR
  %l1 = memory #L1
  %l2 = memory #L2
  %ddrlu = exec_unit #DDRLU
  %l1su = exec_unit #L1SU
  %l1lu = exec_unit #L1LU
  %sfusu = exec_unit #SFUSU
  %l2su = exec_unit #L2SU
  %l2lu = exec_unit #L2LU
  %sfu = exec_unit #SFU

  datapath #PATH %ddr to %ddrlu : memory, exec_unit
  datapath #PATH %ddrlu to %l1 : exec_unit, memory
  datapath #PATH %l1 to %l1su : memory, exec_unit
  datapath #PATH %l1su to %ddr : exec_unit, memory
  datapath #PATH %l1 to %l1lu : memory, exec_unit
  datapath #PATH %l1lu to %sfu : exec_unit, exec_unit
  datapath #PATH %l1lu to %l2su : exec_unit, exec_unit
  datapath #PATH %l2su to %l2 : exec_unit, memory
  datapath #PATH %l2 to %l2lu : memory, exec_unit
  datapath #PATH %l2lu to %sfu : exec_unit, exec_unit
  datapath #PATH %sfu to %sfusu : exec_unit, exec_unit
  datapath #PATH %sfusu to %l1 : exec_unit, memory
}
