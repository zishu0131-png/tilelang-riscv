.section .text.start,"ax"
.global _start
_start:
    csrr t0, mhartid
    bnez t0, .Lpark
    .option push
    .option norelax
    la sp, __stack_top
    la t0, trap_handler
    csrw mtvec, t0
    .option pop
    li t0, 0x6600
    csrs mstatus, t0
    csrw fcsr, zero
    la t0, __bss_start
    la t1, __bss_end
.Lzero:
    bgeu t0, t1, .Lmain
    sd zero, 0(t0)
    addi t0, t0, 8
    j .Lzero
.Lmain:
    call kmain
    li t0, 0x100000
    li t1, 0x13333
    sw t1, 0(t0)
.Lpark:
    wfi
    j .Lpark

.balign 4
trap_handler:
    li t0, 0x10000000
    li t1, 84
    sb t1, 0(t0)
    li t1, 82
    sb t1, 0(t0)
    li t1, 65
    sb t1, 0(t0)
    li t1, 80
    sb t1, 0(t0)
    li t1, 10
    sb t1, 0(t0)
    li t0, 0x100000
    li t1, 0x23333
    sw t1, 0(t0)
    j .Lpark
