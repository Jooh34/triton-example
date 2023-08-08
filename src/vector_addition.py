import torch

import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # 벡터 사이즈.
    BLOCK_SIZE: tl.constexpr,  # 각 프로그램이 처리하는 element 개수.
                 # NOTE: `constexpr` shape 값으로도 사용될 수 있다.
):
    # 다른 데이터를 처리하는 여러 '프로그램'이 존재한다. 여기 실행부가 어떤 프로그램인지
    # 여기서 정의:
    pid = tl.program_id(axis=0)  # 1D 그리드를 사용하기 때문에 axis는 0.

    # 이 프로그램은 초기 데이터로부터 offset만큼 떨어진 input을 처리한다.
    # 예를들어, 256사이즈 벡터와 64사이즈 block_size를 가진다면,
    # 프로그램은 각각 [0:64, 64:128, 128:192, 192:256] 의 element를 접근한다. 
    # offsets은 포인터 리스트임에 주의.
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # out-of-bounds 액세스를 막기위해서 mask를 만든다.
    mask = offsets < n_elements
    # x,y를 DRAM으로부터 로드한다. input이 block size의 배수가 아닐 때를 대비해 extra elements를 마스킹 아웃한다.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # x + y 결과를 DRAM에 저장한다.
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # output을 미리 할당한다.
    output = torch.empty_like(x)
    assert x.is_cuda and y.is_cuda and output.is_cuda
    n_elements = output.numel()
    # SPMD lauch grid는 병렬로 수행되는 커널 인스턴스의 개수이다.
    # CUDA lautnch grid랑 비교할만한데, Tuple[int] 혹은 Callable(metaparamters) -> Tuple[int] 둘 중 하나가 된다.
    # 이 예제에서는 사이즈가 blocks개수인 1D grid이다.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # NOTE:
    #  - 각 torch.tensor 오브젝트는 첫 번째 원소를 가리키는 포인터로 암묵적 변환된다.
    #  - `triton.jit`된 함수는 실행가능한 GPU 커널을 얻기위해 launch grid로 번호매겨진다.
    #  - 키워드 인자로 meta-parameter를 넘기는 것을 잊지않아야한다.
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    
    # z의 핸들을 return받지만, `torch.cuda.synchronize()` 가 호출되지 않으면, 커널은 여전히 이 지점에서 비동기로 실행 중이다.
    return output
    
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
output_torch = x + y
output_triton = add(x, y)
print(output_torch)
print(output_triton)
print(
    f'The maximum difference between torch and triton is '
    f'{torch.max(torch.abs(output_torch - output_triton))}'
)

@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=['size'],  # Argument names to use as an x-axis for the plot.
        x_vals=[
            2 ** i for i in range(12, 28, 1)
        ],  # Different possible values for `x_name`.
        x_log=True,  # x axis is logarithmic.
        line_arg='provider',  # Argument name whose value corresponds to a different line in the plot.
        line_vals=['triton', 'torch'],  # Possible values for `line_arg`.
        line_names=['Triton', 'Torch'],  # Label name for the lines.
        styles=[('blue', '-'), ('green', '-')],  # Line styles.
        ylabel='GB/s',  # Label name for the y-axis.
        plot_name='vector-add-performance',  # Name for the plot. Used also as a file name for saving the plot.
        args={},  # Values for function arguments not in `x_names` and `y_name`.
    )
)
def benchmark(size, provider):
    x = torch.rand(size, device='cuda', dtype=torch.float32)
    y = torch.rand(size, device='cuda', dtype=torch.float32)
    quantiles = [0.5, 0.2, 0.8]
    if provider == 'torch':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: x + y)
    if provider == 'triton':
        ms, min_ms, max_ms = triton.testing.do_bench(lambda: add(x, y))
    gbps = lambda ms: 12 * size / ms * 1e-6
    return gbps(ms), gbps(max_ms), gbps(min_ms)

benchmark.run(print_data=True, show_plots=False)