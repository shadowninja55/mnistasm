default rel
global _start

O_RDONLY equ 0x0
PROT_READ equ 0x1
PROT_WRITE equ 0x2
MAP_PRIVATE equ 0x02
MAP_ANONYMOUS equ 0x20
MAP_POPULATE equ 0x8000

%macro POPCOUNT 0
  ; zmm0 (arg0) = v
  ; rax (result) = ||v||^2
  vpopcntq zmm0, zmm0
  vextracti64x4 ymm1, zmm0, 1
  vpaddq ymm0, ymm0, ymm1
  vextracti128 xmm1, ymm0, 1
  vpaddq xmm0, xmm0, xmm1
  movq rax, xmm0
  vpextrq rbx, xmm0, 1
  add rax, rbx
%endmacro

section .rodata
train_path: db "./train.dat", 0
test_path: db "./test.dat", 0
hundred: db "100.00%", 0

section .bss
train_labels: resq 1
train_data: resq 1
train_inv_norms: resq 1
train_count: resd 1

test_labels: resq 1
test_data: resq 1
test_count: resd 1

section .text
_start:
  ; load training data
  mov rdi, train_path
  mov rsi, train_labels
  mov rdx, train_data
  call load_set

  mov [train_count], eax

  ; calculate training inverse norms
  call inv_norms

  ; load test data
  mov rdi, test_path
  mov rsi, test_labels
  mov rdx, test_data
  call load_set

  ; evaluate
  call evaluate

  ; exit(0)
  mov rax, 60
  mov rdi, 0
  syscall

load_set:
  ; rdi (arg0) = path
  ; rsi (arg1) = labels
  ; rdx (arg2) = data
  ; rax (result) = count
  ;
  ; rbp - 8 = labels
  ; rbp - 16 = data
  ; rbp - 20 = fd
  ; rbp - 24 = count
  push rbp
  mov rbp, rsp
  sub rsp, 16

  mov [rbp - 8], rsi
  mov [rbp - 16], rdx

  ; open(path, flags = O_RDONLY)
  mov rax, 2
  mov rdi, rdi
  mov rsi, O_RDONLY
  syscall

  mov [rbp - 20], eax

  ; read length header
  mov rax, 0
  mov rdi, [rbp - 20]
  lea rsi, [rbp - 24]
  mov rdx, 4
  syscall

  ; mmap(addr = 0, len = 4 + count * 65, prot = PROT_READ, flags = MAP_PRIVATE | MAP_POPULATE, fd, offset = 0)
  mov rax, 9
  mov rdi, 0
  imul esi, [rbp - 24], 65
  add esi, 4
  mov rdx, PROT_READ
  mov r10, MAP_PRIVATE | MAP_POPULATE
  mov r8d, [rbp - 20]
  mov r9, 0
  syscall

  lea rbx, [rax + 4]

  ; close(fd)
  mov rax, 3
  mov rdi, [rbp - 20]
  syscall

  ; labels @ base + 4
  mov rax, [rbp - 8]
  mov [rax], rbx

  ; data @ labels + count
  mov ecx, [rbp - 24]
  add rbx, rcx
  mov rax, [rbp - 16]
  mov [rax], rbx

  ; return count
  mov eax, [rbp - 24]
  leave
  ret

inv_norms:
  ; mmap(addr = 0, len = count * 4, prot = PROT_READ | PROT_WRITE, flags = MAP_PRIVATE | MAP_ANONYMOUS, fd = -1, offset = 0)
  mov rax, 9
  mov rdi, 0
  imul esi, [train_count], 4
  mov rdx, PROT_READ | PROT_WRITE
  mov r10, MAP_PRIVATE | MAP_ANONYMOUS
  mov r8d, -1
  mov r9, 0
  syscall

  mov [train_inv_norms], rax
  mov ecx, 0
.loop:
  mov rax, [train_data]
  lea rbx, [ecx * 8]
  vmovdqu64 zmm0, [rax + rbx * 8]

  POPCOUNT
  vcvtsi2ss xmm0, xmm0, rax
  vrsqrtss xmm0, xmm0, xmm0
  
  mov rax, [train_inv_norms]
  movd [rax + rcx * 4], xmm0

  inc ecx
  cmp ecx, [train_count]
  jl .loop

  ret 

classify:
  ; rdi (arg0) = query
  ; rax (result) = predicted label
  ;
  ; rbp - 4: best label
  ; rbp - 8: best similarity (float)
  push rbp
  mov rbp, rsp
  sub rsp, 8

  mov ecx, 0
  mov qword [rbp - 8], 0
  vmovdqu64 zmm2, [rdi]
.loop:
  mov rax, [train_data]
  lea rbx, [ecx * 8]
  vmovdqu64 zmm1, [rax + rbx * 8]

  vpandq zmm0, zmm1, zmm2 
  POPCOUNT

  vcvtsi2ss xmm0, xmm0, rax
  mov rax, [train_inv_norms]
  movd xmm1, [rax + rcx * 4]
  vmulss xmm0, xmm0, xmm1

  vucomiss xmm0, [rbp - 8]
  jbe .next

  ; better similarity
  movd [rbp - 8], xmm0

  ; better label
  mov rax, [train_labels]
  mov al, [rax + rcx]
  mov [rbp - 4], al
.next:
  inc ecx
  cmp ecx, [train_count]
  jl .loop

  mov eax, [rbp - 4]
  leave
  ret

evaluate:
  ; rbp - 4: i
  ; rbp - 8: count
  ; rbp - 16: output
  push rbp
  mov rbp, rsp
  sub rsp, 16

  mov [test_count], eax
  mov qword [rbp - 8], 0
.loop:
  ; classify(test_data[i])
  mov rbx, [test_data]
  mov ecx, [rbp - 4]
  lea ecx, [ecx * 8]
  lea rdi, [rbx + rcx * 8]
  call classify

  ; put real label into rbx
  mov rbx, [test_labels]
  mov ecx, [rbp - 4]
  mov bl, byte [rbx + rcx]
  movzx rbx, bl
  
  ; inc count if real == predicted
  mov rdx, 0
  cmp rax, rbx
  sete dl
  add [rbp - 8], edx 

  inc dword [rbp - 4]
  mov ecx, [rbp - 4]
  cmp ecx, [test_count]
  jl .loop

  ; calculate acc in (acc / 10_000)
  mov eax, [rbp - 8]
  imul eax, eax, 10_000

  xor edx, edx
  div dword [test_count]

  cmp eax, 10_000
  je .perfect

  ; stringify, TODO: don't assume we won't be 100% accurate...
  mov [rbp - 16 + 6], 10
  mov [rbp - 16 + 5], '%'
  mov [rbp - 16 + 2], '.'

  mov ecx, 10 

%macro DIGIT 1
  xor edx, edx
  div ecx
  add dl, '0'
  mov [rbp - 16 + %1], dl
%endmacro

  DIGIT 4
  DIGIT 3
  DIGIT 1
  DIGIT 0

  ; write(stdout, "xx.xx%\n", 7)
  mov rax, 1
  mov rdi, 1
  lea rsi, [rbp - 16]
  mov rdx, 7
  syscall

  leave
  ret
.perfect:
  ; write(stdout, "100.00%\n", 7)
  mov rax, 1
  mov rdi, 1
  lea rsi, hundred
  mov rdx, 8
  syscall

  leave
  ret
