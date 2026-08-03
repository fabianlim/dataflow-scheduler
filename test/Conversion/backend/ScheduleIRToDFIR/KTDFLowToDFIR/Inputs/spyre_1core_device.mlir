#DDR = {
  kind = "HBM",
  size = 1073741824
}
#DDR_MNILU = {
  ktdf_arch.transfer_granularity = array<i64: 64>
}
#MNISU_DDR = {
  ktdf_arch.transfer_granularity = array<i64: 64>
}
#L1 = {
  kind = "LX",
  size = 1048576
}
#MNILU = {
  kind = "L3LU",
  load_store,
  dataflow_scheduler.double_buffer_last
}
#MNILU_L1 = {
  ktdf_arch.transfer_granularity = array<i64: 64>
}
#MNISU = {
  kind = "L3SU",
  load_store,
  dataflow_scheduler.double_buffer_last
}
#L1_MNISU = {
  ktdf_arch.transfer_granularity = array<i64: 64>
}
#SFU = {
  kind = "SFP",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 64> }
  }
}
#L1LU = {
  kind = "LXLU",
  load_store,
  ktdf_arch.features = { ktdf_arch.feature.simd = { splat, zero_pad } }
}
#L1LU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1LU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1SU = {
  kind = "LXSU",
  load_store
}
#L1SU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1SU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

ktdf_arch.device @spyre_1core attributes {
  mem_space_mapping = #ktdf_arch.map<
    #ktdp.spyre_memory_space<HBM> = "HBM",
    #ktdp.spyre_memory_space<LX>  = "LX"
  >
} {
  %ddr = memory #DDR

  group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU
    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory
    group share(%l1) {
      %sfu = exec_unit #SFU
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
    group share(%l1) {
      %sfu = exec_unit #SFU
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU
      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
  }
}
