// RUN: dataflow-scheduler-opt %s | dataflow-scheduler-opt

// A 4-core ring device. Each core has the sample_device DDR <-> L1 DMA
// units and one corelet with the sample_device L1 -> L1LU -> SFU -> L1SU -> L1
// chain. Every core's MNISU can inject into, and its MNILU can eject from, a
// ring stop. The stops are linked into a bidirectional ring in core order:
// clockwise links go from stop K to stop K+1, counter-clockwise links from
// stop K to stop K-1 (modulo 4).
//
// Ports of each ring stop:
//   0: cw_in    1: ccw_in    2: inject    3: cw_out    4: ccw_out    5: eject
//
// A stop forwards traffic in the direction it arrived in, and may drop a copy
// at its own core on the way (cw_in -> {cw_out, eject}, ccw_in -> {ccw_out,
// eject}). Injected traffic leaves in either direction but never ejects
// locally, and nothing turns around.
//
// The corelet group has a kind, like the core group, so that views that
// flatten groups by kind (the routing graph) treat every corelet as an
// instance of one class rather than as a distinct group of its own.

#DDR = {
  kind = "DDR",
  size = 1073741824 //1GB
}
#DDR_MNILU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU_DDR = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#L1 = {
  kind = "L1",
  size = 1048576 //1MB
}
#MNILU = {
  kind = "MNILU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
  dataflow_scheduler.double_buffer_last
}
#MNILU_L1 = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}
#MNISU = {
  kind = "MNISU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      word_size = #ktdf_arch.map<"DDR" = 64, "L1" = 64>,
      access_granularity = #ktdf_arch.map<
        "DDR" = [{size_in_words = 1, align_in_words = 1}],
        "L1" = [{size_in_words = 1, align_in_words = 1}]
      >
    }
  },
  dataflow_scheduler.double_buffer_last
}
#L1_MNISU = {
  ktdf_arch.transfer_granularity = array<i64: 64> //64B
}

#SFU = {
  kind = "SFU",
  ktdf_arch.features = {
    ktdf_arch.feature.compute,
    ktdf_arch.feature.simd = { lanes = #ktdf_arch.map<f16 = 64> }
  }
}
#SFU_REG = {
  kind = "SFU_REG",
  size = 2048 //2KB
}

#L1LU = {
  kind = "L1LU",
  ktdf_arch.features = {
    ktdf_arch.feature.load = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 64, align_in_words = 64},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    },
    ktdf_arch.feature.simd = { splat, zero_pad }
  }
}
#L1LU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1LU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

#L1SU = {
  kind = "L1SU",
  ktdf_arch.features = {
    ktdf_arch.feature.store = {
      access_granularity = #ktdf_arch.map<
        "L1" = [
          {size_in_words = 64, align_in_words = 64},
          {size_in_words = 2, align_in_words = 2}
        ]
      >
    }
  }
}
#L1SU_CORE_FIFO = {
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}
#L1SU_SUBCORE_FIFO = {
  ktdf_arch.transfer_granularity = array<i64: 64, 2>,
  ktdf_arch.features = { ktdf_arch.feature.queue = { depth = 16, ordered } }
}

#RING_STOP = {
  kind = "RING_STOP",
  connectivity = sparse<[[0, 3], [0, 5], [1, 4], [1, 5], [2, 3], [2, 4]], true> : tensor<6x6xi1>
}
#RING_CW = { kind = "RING_CW" }
#RING_CCW = { kind = "RING_CCW" }
ktdf_arch.device @ring_device {
  %ddr = memory #DDR

  // Core 0.
  %mnilu0, %mnisu0 = group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    group { kind = "corelet" } share(%l1) {
      %sfu = group { kind = "SFU_Block" } share() {
        %sfu_reg = memory #SFU_REG
        %sfu_unit = exec_unit #SFU
        yield %sfu_unit
      } -> exec_unit
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
    yield %mnilu, %mnisu
  } -> exec_unit, exec_unit

  // Core 1.
  %mnilu1, %mnisu1 = group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    group { kind = "corelet" } share(%l1) {
      %sfu = group { kind = "SFU_Block" } share() {
        %sfu_reg = memory #SFU_REG
        %sfu_unit = exec_unit #SFU
        yield %sfu_unit
      } -> exec_unit
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
    yield %mnilu, %mnisu
  } -> exec_unit, exec_unit

  // Core 2.
  %mnilu2, %mnisu2 = group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    group { kind = "corelet" } share(%l1) {
      %sfu = group { kind = "SFU_Block" } share() {
        %sfu_reg = memory #SFU_REG
        %sfu_unit = exec_unit #SFU
        yield %sfu_unit
      } -> exec_unit
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
    yield %mnilu, %mnisu
  } -> exec_unit, exec_unit

  // Core 3.
  %mnilu3, %mnisu3 = group { kind = "core" } share(%ddr) {
    %l1 = memory #L1
    %mnilu = exec_unit #MNILU
    %mnisu = exec_unit #MNISU

    datapath #DDR_MNILU %ddr to %mnilu : memory, exec_unit
    datapath #MNILU_L1 %mnilu to %l1 : exec_unit, memory
    datapath #L1_MNISU %l1 to %mnisu : memory, exec_unit
    datapath #MNISU_DDR %mnisu to %ddr : exec_unit, memory

    group { kind = "corelet" } share(%l1) {
      %sfu = group { kind = "SFU_Block" } share() {
        %sfu_reg = memory #SFU_REG
        %sfu_unit = exec_unit #SFU
        yield %sfu_unit
      } -> exec_unit
      %l1lu = exec_unit #L1LU
      %l1su = exec_unit #L1SU

      datapath #L1LU_CORE_FIFO %l1 to %l1lu : memory, exec_unit
      datapath #L1LU_SUBCORE_FIFO %l1lu to %sfu : exec_unit, exec_unit
      datapath #L1SU_SUBCORE_FIFO %sfu to %l1su : exec_unit, exec_unit
      datapath #L1SU_CORE_FIFO %l1su to %l1 : exec_unit, memory
    }
    yield %mnilu, %mnisu
  } -> exec_unit, exec_unit

  // Ring stops, one per core.
  %s0:6 = switch[6] #RING_STOP
  %s1:6 = switch[6] #RING_STOP
  %s2:6 = switch[6] #RING_STOP
  %s3:6 = switch[6] #RING_STOP

  // Each core injects into, and ejects from, its own stop.
  datapath %mnisu0 to %s0#2 : exec_unit, port
  datapath %s0#5 to %mnilu0 : port, exec_unit
  datapath %mnisu1 to %s1#2 : exec_unit, port
  datapath %s1#5 to %mnilu1 : port, exec_unit
  datapath %mnisu2 to %s2#2 : exec_unit, port
  datapath %s2#5 to %mnilu2 : port, exec_unit
  datapath %mnisu3 to %s3#2 : exec_unit, port
  datapath %s3#5 to %mnilu3 : port, exec_unit

  // Clockwise ring: stop K -> stop K+1.
  datapath #RING_CW %s0#3 to %s1#0 : port, port
  datapath #RING_CW %s1#3 to %s2#0 : port, port
  datapath #RING_CW %s2#3 to %s3#0 : port, port
  datapath #RING_CW %s3#3 to %s0#0 : port, port

  // Counter-clockwise ring: stop K -> stop K-1.
  datapath #RING_CCW %s0#4 to %s3#1 : port, port
  datapath #RING_CCW %s1#4 to %s0#1 : port, port
  datapath #RING_CCW %s2#4 to %s1#1 : port, port
  datapath #RING_CCW %s3#4 to %s2#1 : port, port
}
