import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import matplotlib.pyplot as plt
from matplotlib.tri import Triangulation

# 參數設定
eps = 0.01
beta = np.array([1.0, 1.0])
T = 1.0

def u_exact(x, y, t):
    return np.exp(-t) * np.sin(np.pi*x) * np.sin(np.pi*y)

def f_source(x, y, t):
    pi = np.pi
    et = np.exp(-t)
    term1 = (0.02*pi*pi - 1) * np.sin(pi*x)*np.sin(pi*y)
    term2 = pi * (np.cos(pi*x)*np.sin(pi*y) + np.sin(pi*x)*np.cos(pi*y))
    return et * (term1 + term2)

def create_mesh(Nx, Ny):
    """產生單位矩形上的結構化三角網格"""
    x = np.linspace(0, 1, Nx+1)
    y = np.linspace(0, 1, Ny+1)
    X, Y = np.meshgrid(x, y)
    nodes = np.vstack([X.ravel(), Y.ravel()]).T
    elements = []
    for j in range(Ny):
        for i in range(Nx):
            n0 = j*(Nx+1) + i
            n1 = j*(Nx+1) + i+1
            n2 = (j+1)*(Nx+1) + i
            n3 = (j+1)*(Nx+1) + i+1
            elements.append([n0, n1, n2])
            elements.append([n1, n3, n2])
    return nodes, np.array(elements)

def is_boundary_node(node, tol=1e-12):
    x, y = node
    return abs(x-0)<tol or abs(x-1)<tol or abs(y-0)<tol or abs(y-1)<tol

def assemble_matrices(nodes, elements):
    """組裝質量矩陣 M、擴散剛度矩陣 K (含 eps)、對流矩陣 C"""
    N_nodes = nodes.shape[0]
    N_elem = elements.shape[0]
    M = sp.lil_matrix((N_nodes, N_nodes))
    K = sp.lil_matrix((N_nodes, N_nodes))
    C = sp.lil_matrix((N_nodes, N_nodes))
    
    for elem in elements:
        coords = nodes[elem]
        # 單元面積
        A = 0.5 * abs(np.linalg.det(np.vstack([coords[1]-coords[0], coords[2]-coords[0]])))
        # 局部質量矩陣
        M_local = A/12.0 * (np.eye(3) + 1)
        # 計算梯度矩陣 (2x3)
        D = np.ones((3,3))
        D[:,1:] = coords
        coeff = np.linalg.inv(D)
        grad_phi = coeff[1:, :].T  # 3x2
        # 局部擴散矩陣 (含 eps)
        K_local = eps * A * (grad_phi @ grad_phi.T)
        # 局部對流矩陣
        C_local = np.zeros((3,3))
        for i in range(3):
            for j in range(3):
                C_local[i,j] = (beta @ grad_phi[j,:]) * (A/3.0)
        # 組裝
        for a in range(3):
            ii = elem[a]
            for b in range(3):
                jj = elem[b]
                M[ii, jj] += M_local[a,b]
                K[ii, jj] += K_local[a,b]
                C[ii, jj] += C_local[a,b]
    return M.tocsr(), K.tocsr(), C.tocsr()

def assemble_K0(nodes, elements):
    """組裝不含 eps 的剛度矩陣 K0 (用於 H1 半範數)"""
    N_nodes = nodes.shape[0]
    K0 = sp.lil_matrix((N_nodes, N_nodes))
    for elem in elements:
        coords = nodes[elem]
        A = 0.5 * abs(np.linalg.det(np.vstack([coords[1]-coords[0], coords[2]-coords[0]])))
        D = np.ones((3,3))
        D[:,1:] = coords
        coeff = np.linalg.inv(D)
        grad_phi = coeff[1:, :].T
        K0_local = A * (grad_phi @ grad_phi.T)
        for a in range(3):
            ii = elem[a]
            for b in range(3):
                jj = elem[b]
                K0[ii, jj] += K0_local[a,b]
    return K0.tocsr()

def assemble_load(nodes, elements, t):
    """組裝載荷向量 f(t)"""
    N_nodes = nodes.shape[0]
    F = np.zeros(N_nodes)
    for elem in elements:
        coords = nodes[elem]
        centroid = np.mean(coords, axis=0)
        f_val = f_source(centroid[0], centroid[1], t)
        A = 0.5 * abs(np.linalg.det(np.vstack([coords[1]-coords[0], coords[2]-coords[0]])))
        for i, node in enumerate(elem):
            F[node] += f_val * (A/3.0)
    return F

def crank_nicolson(Nx, Ny, dt, T):
    """執行 Crank-Nicolson 時間推進，回傳最終節點解及誤差"""
    nodes, elements = create_mesh(Nx, Ny)
    N_nodes = nodes.shape[0]
    M, K, C = assemble_matrices(nodes, elements)
    K0 = assemble_K0(nodes, elements)   # 用於 H1 誤差
    
    boundary_dofs = [i for i, node in enumerate(nodes) if is_boundary_node(node)]
    interior_dofs = [i for i in range(N_nodes) if i not in boundary_dofs]
    
    A = M + 0.5*dt * (K + C)
    B = M - 0.5*dt * (K + C)
    
    # 施加 Dirichlet 邊界條件 (齊次)
    for dof in boundary_dofs:
        A[dof, :] = 0
        A[dof, dof] = 1.0
        B[dof, :] = 0
        # B[dof, dof] = 1.0  # 可選，但右端會強制為0
    
    # 初始條件
    u = np.array([u_exact(node[0], node[1], 0.0) for node in nodes])
    for dof in boundary_dofs:
        u[dof] = 0.0
    
    n_steps = int(T / dt)
    for n in range(n_steps):
        t_n = n * dt
        t_np1 = (n+1) * dt
        f_n = assemble_load(nodes, elements, t_n)
        f_np1 = assemble_load(nodes, elements, t_np1)
        rhs = B @ u + 0.5*dt * (f_n + f_np1)
        for dof in boundary_dofs:
            rhs[dof] = 0.0
        u = spla.spsolve(A, rhs)
    
    # 計算最終誤差
    u_ex_vec = np.array([u_exact(node[0], node[1], T) for node in nodes])
    e = u - u_ex_vec
    L2_err = np.sqrt(e @ M @ e)
    H1_err = np.sqrt(e @ K0 @ e)
    return nodes, u, L2_err, H1_err

def spatial_convergence_test():
    N_list = [8, 16, 32, 64]
    h_list = [1.0/N for N in N_list]
    dt_list = [h*h for h in h_list]
    L2_errs = []
    H1_errs = []
    for N, dt in zip(N_list, dt_list):
        print(f"空間測試: N={N}, dt={dt:.2e}")
        _, _, L2, H1 = crank_nicolson(N, N, dt, T)
        L2_errs.append(L2)
        H1_errs.append(H1)
    rates_L2 = [np.log(L2_errs[i-1]/L2_errs[i])/np.log(2) for i in range(1, len(L2_errs))]
    rates_H1 = [np.log(H1_errs[i-1]/H1_errs[i])/np.log(2) for i in range(1, len(H1_errs))]
    print("\n空間收斂表:")
    print("N\t h\t L2 error\t L2 rate\t H1 error\t H1 rate")
    for i, N in enumerate(N_list):
        l2r = f"{rates_L2[i-1]:.2f}" if i>0 else '-'
        h1r = f"{rates_H1[i-1]:.2f}" if i>0 else '-'
        print(f"{N}\t {1/N:.4f}\t {L2_errs[i]:.2e}\t {l2r}\t\t {H1_errs[i]:.2e}\t {h1r}")
    return N_list, L2_errs, H1_errs

def temporal_convergence_test():
    N_fine = 128
    dt_list = [1.0/10, 1.0/20, 1.0/40, 1.0/80]
    L2_errs = []
    for dt in dt_list:
        print(f"時間測試: dt={dt:.2e}, N={N_fine}")
        _, _, L2, _ = crank_nicolson(N_fine, N_fine, dt, T)
        L2_errs.append(L2)
    rates = [np.log(L2_errs[i-1]/L2_errs[i])/np.log(2) for i in range(1, len(L2_errs))]
    print("\n時間收斂表:")
    print("dt\t L2 error\t rate")
    for i, dt in enumerate(dt_list):
        r = f"{rates[i-1]:.2f}" if i>0 else '-'
        print(f"{dt}\t {L2_errs[i]:.2e}\t {r}")
    return dt_list, L2_errs

def plot_results():
    # 求解精細結果供繪圖 (N=64, dt=1/4096)
    nodes, u_num, _, _ = crank_nicolson(64, 64, 1/4096, T)
    u_ex = np.array([u_exact(node[0], node[1], T) for node in nodes])
    _, elements = create_mesh(64, 64)
    tri = Triangulation(nodes[:,0], nodes[:,1], elements)
    
    fig = plt.figure(figsize=(12,5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_trisurf(tri, u_num, cmap='viridis')
    ax1.set_title('Numerical solution at T=1')
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_trisurf(tri, u_ex, cmap='plasma')
    ax2.set_title('Exact solution at T=1')
    plt.show()
    
    # 空間收斂曲線
    N_list, L2_errs, H1_errs = spatial_convergence_test()
    h_list = [1.0/N for N in N_list]
    plt.figure()
    plt.loglog(h_list, L2_errs, 'o-', label='L2 error')
    plt.loglog(h_list, H1_errs, 's-', label='H1 error')
    plt.loglog(h_list, [h**2 for h in h_list], 'k--', label='O(h^2)')
    plt.loglog(h_list, [h for h in h_list], 'k:', label='O(h)')
    plt.xlabel('h')
    plt.ylabel('Error')
    plt.legend()
    plt.title('Spatial convergence')
    plt.grid(True)
    plt.show()
    
    # 時間收斂曲線
    dt_list, L2_temporal = temporal_convergence_test()
    plt.figure()
    plt.loglog(dt_list, L2_temporal, 'o-', label='L2 error')
    plt.loglog(dt_list, [dt**2 for dt in dt_list], 'k--', label='O(Δt^2)')
    plt.xlabel('Δt')
    plt.ylabel('L2 error')
    plt.legend()
    plt.title('Temporal convergence')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    # 執行收斂測試及繪圖
    plot_results()