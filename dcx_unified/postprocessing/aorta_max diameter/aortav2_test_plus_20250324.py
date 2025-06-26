import os
import nibabel
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter, gaussian_filter1d, distance_transform_edt
from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
import time
from cv2 import connectedComponentsWithStats
import cv2
from skimage.transform import rescale
from matplotlib.collections import LineCollection
import sys
from skimage.feature import hessian_matrix, hessian_matrix_eigvals

from scipy import odr

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import numpy as np
from sklearn.linear_model import Ridge
from matplotlib import pyplot as plt
from scipy import sparse
from scipy.optimize import minimize, lsq_linear
from matplotlib.collections import LineCollection

def cubic_formula(a, b, c, d):
    delta_0 = (b**2 - 3*a*c).astype('complex')
    delta_1 = (2*b**3 - 9*a*b*c + 27*a**2*d).astype('complex')
    
    C = ((delta_1 + (delta_1**2-4*delta_0**3)**(1/2))/2)**(1/3)
    C123 = C * np.array([1, (-1 + (-3)**0.5)/2,  (-1 - (-3)**0.5)/2]).reshape(-1,1)
    roots = -1/(3*a) * (b + C123 + np.divide(delta_0,C123, out=np.zeros_like(C123), where= C123!=0))
    return roots

def find_delta_epsilon(alpha, gamma, a, b, c):
    cubic_coefs = 2 * a**2, 3*a*b, b**2 + 2*a*(c-gamma)+1, b*(c-gamma)- alpha
    print("a_coeff : ", 2 * a**2)
    print("b_coeff : ", 3*a*b)
    print("c_coeff : ", sum(b**2 + 2*a*(c-gamma)+1))
    print("d_coeff : ", sum(b*(c-gamma)- alpha))
    orthogonal_xs = cubic_formula(*cubic_coefs)
    orthogonal_ys = a * orthogonal_xs**2 + b * orthogonal_xs + c

    print("orthogonal_xs sum : ", np.sum(orthogonal_xs))
    print("orthogonal_ys sum : ", np.sum(orthogonal_ys))

    real_xs = np.where(np.isclose(orthogonal_xs, np.real(orthogonal_xs)), np.real(orthogonal_xs), np.infty)
    real_ys = np.where(np.isclose(orthogonal_xs, np.real(orthogonal_xs)), np.real(orthogonal_ys), np.infty)

    
    closest_idx = np.argmin((real_xs - np.real(alpha))**2 + (real_ys - np.real(gamma))**2, axis=0)
    closest_x = real_xs[closest_idx, np.arange(len(real_xs[0]))]
    closest_y = real_ys[closest_idx, np.arange(len(real_ys[0]))]

    print("closest_idx sum : ", np.sum(closest_idx))
    print("closest_x sum : ", np.sum(closest_x))
    print("closest_y sum : ", np.sum(closest_y))
    
#     alpha_inf = orthogonal_xs[:,np.where(np.isinf(closest_x))]
#     gamma_inf = orthogonal_ys[:,np.where(np.isinf(closest_y))]

#     print(alpha_inf, gamma_inf)
    return np.real(alpha) - closest_x, np.real(gamma) - closest_y
def delta_partial_a(delta, alpha, gamma, a, b, c):
    x_ = (alpha-delta)
    _epsilon = a * x_**2 + b*x_ + c - gamma
    tangent = 2*a*x_ + b
    return (2 * x_ * _epsilon + tangent * x_**2) / (2*a*_epsilon + tangent**2 + 1)

def delta_partial_b(delta, alpha, gamma, a, b, c):
    x_ = (alpha-delta)
    _epsilon = a * x_**2 + b*x_ + c - gamma
    tangent = 2*a*x_ + b
    
    return (_epsilon + tangent * x_) / (2*a*_epsilon + tangent**2 + 1)

def delta_partial_c(delta, alpha, gamma, a, b, c):
    x_ = (alpha-delta)
    _epsilon = a * x_**2 + b*x_ + c - gamma
    tangent = 2*a*x_ + b
    
    return tangent / (2*a*_epsilon + tangent**2 + 1)


'''
Original version
'''
# def g_partial_a(delta, alpha, gamma, a, b, c):
#     x_ = (alpha-delta)
#     epsilon = a * x_**2 + b*x_ + c - gamma
#     tangent = 2*a*x_ + b
    
#     delta_a = delta_partial_a(delta, alpha, gamma, a, b, c)
#     return np.sum(2*epsilon*(x_**2 -2*a*x_*delta_a - b * delta_a) + 2 * delta * delta_a)
    
    
# def g_partial_b(delta, alpha, gamma, a, b, c):
#     x_ = (alpha-delta)
#     epsilon = a * x_**2 + b*x_ + c - gamma
#     tangent = 2*a*x_ + b
    
#     delta_b = delta_partial_b(delta, alpha, gamma, a, b, c)
#     return np.sum(2*epsilon*(-2*a*x_*delta_b +x_ - b * delta_b) + 2*delta*delta_b)

# def g_partial_c(delta, alpha, gamma, a, b, c):
#     x_ = (alpha-delta)
#     epsilon = a * x_**2 + b*x_ + c - gamma
#     tangent = 2*a*x_ + b
    
#     delta_c = delta_partial_c(delta, alpha, gamma, a, b, c)
#     return np.sum(2*epsilon*(-2*a*x_*delta_c - b * delta_c + 1) +2*delta*delta_c)


def g_i_partial_a(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    x_ = (xs-deltas)
    _epsilon = a * x_**2 + b*x_ + c - ys
    tangent = 2*a*x_ + b
    
    delta_a = delta_partial_a(deltas, xs, ys, a, b, c)
    return -ws * (x_**2 -2*a*x_*delta_a - b * delta_a)
    
    
def g_i_partial_b(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    x_ = (xs-deltas)
    _epsilon = a * x_**2 + b*x_ + c - ys
    tangent = 2*a*x_ + b
    
    delta_b = delta_partial_b(deltas, xs, ys, a, b, c)
    return -ws * (-2*a*x_*delta_b +x_ - b * delta_b)

def g_i_partial_c(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    x_ = (xs-deltas)
    _epsilon = a * x_**2 + b*x_ + c - ys
    tangent = 2*a*x_ + b
    
    delta_c = delta_partial_c(deltas, xs, ys, a, b, c)
    return -ws * (-2*a*x_*delta_c - b * delta_c + 1) 

def g_in_partial_a(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    delta_a = delta_partial_a(deltas, xs, ys, a, b, c)

    return ds * delta_a
    
def g_in_partial_b(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    delta_b = delta_partial_b(deltas, xs, ys, a, b, c)

    return ds * delta_b

def g_in_partial_c(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    delta_c = delta_partial_c(deltas, xs, ys, a, b, c)

    return ds * delta_c

def g_i(beta, deltas, xs, ys, ws, ds):
    a, b, c = beta[0], beta[1], beta[2]
    epsilons = ys - (a * (xs-deltas)**2 + b*(xs-deltas) + c)
    
    return ws * epsilons

def g_in(beta, deltas, xs, ys, ws, ds):
    
    return ds * deltas

def g(beta, deltas, xs, ys, ws, ds):
    n = len(xs)
    a, b, c = beta[0], beta[1], beta[2]
    return np.concatenate((g_i(beta, deltas, xs, ys, ws, ds), g_in(beta, deltas, xs, ys, ws, ds)))
    
def J(beta, deltas, xs, ys, ws, ds):
    jacobian = np.zeros(shape=(2 * len(xs), len(beta)))
    n = len(xs)
    jacobian[:n,0] = g_i_partial_a(beta, deltas, xs, ys, ws, ds)
    jacobian[:n,1] = g_i_partial_b(beta, deltas, xs, ys, ws, ds)
    jacobian[:n,2] = g_i_partial_c(beta, deltas, xs, ys, ws, ds)
    
    jacobian[n:,0] = g_in_partial_a(beta, deltas, xs, ys, ws, ds)
    jacobian[n:,1] = g_in_partial_b(beta, deltas, xs, ys, ws, ds)
    jacobian[n:,2] = g_in_partial_c(beta, deltas, xs, ys, ws, ds)
    
    return jacobian

    

'''
New version
'''

# def g(beta, deltas, xs, ys, ws, ds):
#     a, b, c = beta[0], beta[1], beta[2]
#     epsilons = a * (xs-deltas)**2 + b*(xs-deltas) + c - ys
#     return (ws * epsilons)**2 + (ws*ds*deltas)**2

# def grad(g_array, beta, deltas, xs, ys, ws, ds):
#     grad = np.zeros(len(beta))
    
#     grad[0] = g_partial_a(beta, deltas, xs, ys, ws, ds)
#     grad[1] = g_partial_b(beta, deltas, xs, ys, ws, ds)
#     grad[2] = g_partial_c(beta, deltas, xs, ys, ws, ds)

#     return grad

# def hessian(g_array, beta, deltas, xs, ys, ws, ds):
    
def get_ws(epsilon, delta):
    return 1
    return (1/np.std(epsilon))**2 

def get_ds(epsilon, delta):
    return 1
    return (1/np.std(delta))**2 

def quadratic_odr(init_beta, xs, ys, method, weights=1):
    xs, ys = np.array(xs), np.array(ys)

    if method == "optimizer":
        def objective_function(beta):
            delta, epsilon = find_delta_epsilon(xs, ys, *beta)
            ws = get_ws(ys, xs)
            ds = get_ds(ys, xs)
            return np.sum(g(beta, delta, xs, ys, ws, ds)**2)

        def jac(beta):
            delta, epsilon = find_delta_epsilon(xs, ys, *beta)
            ws = get_ws(ys, xs)
            ds = get_ds(ys, xs)
            jac_a = np.sum(2 * ws * epsilon * g_i_partial_a(beta, delta, xs, ys, ws, ds) + 2 * ds * delta * g_in_partial_a(beta, delta, xs, ys, ws, ds) )
            jac_b = np.sum(2 * ws * epsilon * g_i_partial_b(beta, delta, xs, ys, ws, ds) + 2 * ds * delta * g_in_partial_b(beta, delta, xs, ys, ws, ds) )
            jac_c = np.sum(2 * ws * epsilon * g_i_partial_c(beta, delta, xs, ys, ws, ds) + 2 * ds * delta * g_in_partial_c(beta, delta, xs, ys, ws, ds) )

            return np.array([jac_a, jac_b, jac_c])

        final_beta = minimize(objective_function, init_beta, jac=jac)
        # print(final_beta.x)
        return final_beta.x, None
    
    else:
        
        init_lambd = 1
        lambd_increase = 1000
        lambd_decrease = 2
        # assuming float64
        eps = np.finfo(np.float64).eps
        sstol = eps**0.5
        partol = eps**(2/3)
        max_iter = 10000
        
        beta = init_beta
        delta, epsilon = find_delta_epsilon(xs, ys, *beta)

        # plt.hist(delta, alpha=0.5, label='delta')
        # plt.hist(epsilon, alpha=0.5, label='epsilon')
        # plt.legend()
        # plt.show()
        # sys.exit()
        ws = get_ws(ys, xs) #1
        ds = get_ds(ys, xs) #1

        lambd = init_lambd
        #tau = init_tau
        max_norm = None
        
        iter_count = 0

        prev_list = []
        new_list = []

        lambda_list = []
        region_list = []

        while True:
            print("<<<<<< iteration : ",iter_count)
            
            g_c = g(beta, delta, xs, ys, ws, ds)
            print("g_c sum : ", np.sum(g_c))
            
            g_c = g_c.reshape(-1, 1)
            g_norm = np.sum((g_c)**2)
            
            J_c = J(beta, delta, xs, ys, ws, ds)
            print("J_c sum : ", np.sum(J_c))
            
            ridge_model = Ridge(lambd, solver = 'cholesky', fit_intercept=False,tol=sstol)
            #np.save("-J_c.npy", -J_c) #save
            #np.save("g_c.npy", g_c) #save
            ridge_model.fit(-J_c, g_c)
            s = ridge_model.coef_
            new_beta = beta + s.flatten()
            print("new_beta : ", new_beta)
            
            new_delta, new_epsilon = find_delta_epsilon(xs, ys, *new_beta)
            print("new_delta sum : ", np.sum(new_delta))
            print("epsilon sum : ", np.sum(new_epsilon))
            
            new_g = g(new_beta, new_delta, xs, ys, ws, ds)
            
            print("new_g sum : ", np.sum(new_g))

            new_g_norm = np.sum((new_g)**2)
            prev_list.append(g_norm)
            new_list.append(new_g_norm)
            
            if method.startswith("trust region"):
                expected_g_norm = np.sum((g_c + np.matmul(J_c, s.reshape(-1,1)))**2)
                s_norm = np.sum(s**2)
                print("s_norm:",s_norm)
                if s_norm < partol or np.abs(g_norm - expected_g_norm) < sstol or iter_count >= max_iter:
#                     plt.close()
#                     fig, ax = plt.subplots(1,2)
#                     ax[0].plot(np.arange(len(lambda_list)), lambda_list, "ko")
#                     ax[1].plot(np.arange(len(region_list)), region_list, "ko")

#                     plt.show()
                    
                    return beta
                else:
                    ratio = (g_norm - new_g_norm) / (g_norm - expected_g_norm)
                    print("ratio : ", ratio)
                    if ratio < 1/4:
                        lambd *= lambd_increase
                        max_norm = None
                        print(lambd)
                    else:
                            
                        if ratio >3/4 and max_norm is not None and s_norm >= max_norm:
                            lambd /= lambd_decrease
                            max_norm = None
                            print(lambd)
                        else:
                            if max_norm is None:
                                max_norm = s_norm
                            else:
                                max_norm = max(max_norm, s_norm)
                            beta = new_beta
                            delta = new_delta
                            print("beta : ", beta)
                            print("delta : ", delta)
                iter_count += 1
                


RIDGE_GRAD = 0.3
STANDARD_DIM = 2048


""" 
[Ascending Aorta Only]
Orthogonal distance regression (quadratic regression)
"""
def polynomial_fit(mask, scale_factor=1):
    
    largest_mask = mask.astype("uint8")
    if scale_factor == 1:
        points = np.where(largest_mask)
    else:
        rescaled_mask = (rescale(largest_mask, scale=(1,scale_factor)) > 0).astype('uint8')
        points = np.where(rescaled_mask)

    h_range = np.max(points[0]) - np.min(points[0])
    v_range = np.max(points[1]) - np.min(points[1])

    print("v_range / h_range")
    print(v_range / h_range)
    if v_range / h_range < 3:
        print("return non")
        return None, None, None, None
        
    def fit_func(p, t):
        return p[0] * t**2 + p[1] * t + p[2]
    # 2차 함수 일반 피팅
    fit_ = np.polyfit(points[1], points[0], 2)

    # 위 피팅 결과를 initial point로 하여 orthogonal regression 
    # Model = odr.Model(fit_func)
    # Data = odr.RealData(points[1], points[0])
    # Odr = odr.ODR(Data, Model, fit_, maxit=10000)
    # output= Odr.run()
    # beta = output.beta
    
    beta = quadratic_odr(fit_, points[1], points[0], method='trust region')
    
    
    beta = [beta[0] * scale_factor ** 2, beta[1] * scale_factor, beta[2]]
    # print(beta)
    points = np.where(largest_mask)
    
    
    
    # skeleton 앞 뒤 제거를 위한 edt 및 gradient 계산 
    edt_image = distance_transform_edt(largest_mask)
    grad_x, grad_y = np.gradient(edt_image)

    mag_grad = (grad_x**2 + grad_y**2)**0.5

    # 2차 함수를 사용하여 skeleton 구하기  
    new_y = np.arange(np.min(points[1]), np.max(points[1]))
    s_x, s_y = fit_func(beta, new_y), new_y

    skeleton_x, skeleton_y = [], []
    
    if len(s_x) == 0 or len(s_y) == 0:
        raise ValueError

    # 위 (aortic arch)쪽으로는 함수 방향과 gradient 사이 dot product로 판별 
    first_i, last_i = None, None
    for i, (x, y) in enumerate(zip(s_x, s_y)):
        if inside_mask((x,y), largest_mask):
            if first_i is None:
                first_i = i
                real_first_i = i
            slope = 2 * beta[0] * y + beta[1]
            tan_vector = slope / (slope**2 + 1)**0.5, 1 / (slope**2 + 1)**0.5
            dot_prod = grad_x[round(x), round(y)] * tan_vector[0] + grad_y[round(x), round(y)] * tan_vector[1]
            # print('first', dot_prod)
            if np.abs(dot_prod) < 0.5:
                first_i = i
                break
    
    # 아래 (aortic root) 쪽으로는 함수 방향과 gradient 사이 dot product + mag_grad로 판별 
    last_i_mask, last_i_dot = None, None # First i in mask, First i with dot_prod 
    for i, (x, y) in enumerate(zip(s_x[::-1], s_y[::-1])):
        if inside_mask((x,y), largest_mask):
            if last_i_mask is None:
                last_i_mask = i
                real_last_i = i
            slope = 2 * beta[0] * y + beta[1]
            tan_vector = slope / (slope**2 + 1)**0.5, 1 / (slope**2 + 1)**0.5
            dot_prod = grad_x[round(x), round(y)] * tan_vector[0] + grad_y[round(x), round(y)] * tan_vector[1]
            # print('last', dot_prod, mag_grad[round(x), round(y)])
            if np.abs(dot_prod) < 0.5:
                if last_i_dot is None:
                    last_i_dot = i
                if mag_grad[round(x), round(y)] <0.5:
                    last_i = i
                    break
    if last_i is None:
        if last_i_dot is None:
            if last_i_mask is None:
                last_i = len(s_x)-1
            else:
                last_i = len(s_x)-1 -last_i_mask
        else:
            last_i = len(s_x)-1 - last_i_dot
    else:
        last_i = len(s_x)-1 - last_i_dot
    
    
    skeleton_x, skeleton_y = s_x[first_i:last_i+1], s_y[first_i:last_i+1]
    
    first_point = np.array([s_x[real_first_i], s_y[real_first_i]])
    last_point = np.array([s_x[len(s_x)-1-real_last_i], s_y[len(s_y)-1-real_last_i]])
    print(len(skeleton_x))
    if len(skeleton_x) < 3:
        skeleton_x = s_x
        skeleton_y = s_y
    return np.vstack((skeleton_x, skeleton_y)), first_point, last_point, beta

""" 
[Descending Aorta Only]
Erode + Euclidean distance transform and gradient
Find skeleton candidates (0.3 is the arbitrary threshold for |gradient| of EDT)
"""
def skeleton_candidates(mask):
    mask = mask.astype("uint8")
    
    
    kernel = np.ones(shape=(74,74))
    smoothed_mask = cv2.erode(mask, kernel, iterations=1)
    

    edt_image = distance_transform_edt(smoothed_mask)

    grad_x, grad_y = np.gradient(edt_image)
    mag_grad = (grad_x**2 + grad_y**2)**0.5
    
    skeleton = np.where((smoothed_mask == 1) * (mag_grad < RIDGE_GRAD))
    
    return skeleton

"""
Fill in skeleton using interpolation (if skeleton is vertical, might introduce problem since slope will be undefined) 
"""
def connect_skeleton_candidates(skeleton_set, image):
    ordered_skeleton = np.array(sorted(np.transpose(skeleton_set), key = lambda x: x[1]))
    
    skeleton_x = np.transpose(ordered_skeleton)[0]
    skeleton_y = np.transpose(ordered_skeleton)[1]

    f = interp1d(skeleton_y, skeleton_x)
    new_skeleton_y = np.arange(np.min(skeleton_y), np.max(skeleton_y))
    new_skeleton_x = f(new_skeleton_y)
    
    return np.vstack((new_skeleton_x, new_skeleton_y))


def inside_mask(point, image):
    x, y = point
    return (0 <= round(x) < image.shape[0]) and (0 <=round(y) < image.shape[1]) and (image[round(x),round(y)])


"""
[Deprecated]
Use slope of skeleton to extend endpoints
"""
def extend_up_down(skeleton, image):
    
    skeleton_x = list(skeleton[0])
    skeleton_y = list(skeleton[1])
    # for i, (ske_x, ske_y) in enumerate(zip(skeleton_x, skeleton_y)):
    #     if image[int(round(ske_x)), int(round(ske_y))] == 0:
    #         skeleton_x = list(skeleton_x[:i])
    #         skeleton_y = list(skeleton_y[:i])
    #         break

    x, y = skeleton_x[0], skeleton_y[0]

    # Optional second order estimation, but with interpolation, first order is sufficient
    # x_dir = (3 * x - 4 * skeleton_x[1] + skeleton_x[2]) / 2
    # y_dir = (3 * y - 4 * skeleton_y[1] + skeleton_y[2]) / 2

    # # extend left
    # x_dir = x - skeleton_x[1]
    # y_dir = y - skeleton_y[1]
    # mag = (x_dir**2 + y_dir**2)**0.5
    # x_dir /= mag 
    # y_dir /= mag 
    # while inside_mask((x,y), image):
    #     x += x_dir
    #     y += y_dir
    #     skeleton_x.insert(0, x)
    #     skeleton_y.insert(0, y)
    # x -= x_dir
    # y -= y_dir
    # skeleton_x.pop(0)
    # skeleton_y.pop(0)

    # extend right
    x, y = skeleton_x[-1], skeleton_y[-1]
    x_dir = x - skeleton_x[-2]
    y_dir = y - skeleton_y[-2]
    mag = (x_dir**2 + y_dir**2)**0.5
    x_dir /= mag 
    y_dir /= mag 
    while inside_mask((x,y), image):
        x += x_dir
        y += y_dir
        skeleton_x.append(x)
        skeleton_y.append(y)
    x -= x_dir
    y -= y_dir
    skeleton_x.pop()
    skeleton_y.pop()
    
    return np.vstack((skeleton_x, skeleton_y))

def smooth_gaussian(skeleton):
    skeleton_x = skeleton[0]
    skeleton_y = skeleton[1]
    
    skeleton_x = gaussian_filter1d(skeleton_x, 20, mode='nearest')
    skeleton_y = gaussian_filter1d(skeleton_y, 20, mode='nearest')
    
    return np.vstack((skeleton_x, skeleton_y))
'''
[Deprecated]
Use derivatives to cut-off ascending skeleton
'''

def check_curve(skeleton):
    skeleton_x, skeleton_y = skeleton[0], skeleton[1]

    # y: arch to root direction
    dy = skeleton_y[1:] - skeleton_y[:-1]
    dx = skeleton_x[1:] - skeleton_x[:-1]

    deriv_x = dx / dy
    deriv_y = (skeleton_y[1:] + skeleton_y[:-1])/2

    deriv2_x = (deriv_x[1:] - deriv_x[:-1]) / (deriv_y[1:] - deriv_y[:-1])
    deriv2_y = (deriv_y[1:] + deriv_y[:-1]) / 2
    
    try:
        critical_i = np.where((deriv_x[1:] * deriv_x[:-1] <= 0) & (deriv2_x > 0))[0][0]
        critical_y = deriv_y[critical_i]


        inflection_i = np.where((deriv2_y > critical_y) & (deriv2_x < -5e-3))[0][0]
        cutoff_i = np.where(skeleton_y > deriv2_y[inflection_i])[0][0]


        if cutoff_i > 0:
            skeleton_x = skeleton_x[:cutoff_i]
            skeleton_y = skeleton_y[:cutoff_i]
    except IndexError:
        pass
            
    return np.vstack((skeleton_x, skeleton_y))

'''
Quadratic formula: getting roots of quadratic equation
'''
def quadratic_formula(a_array,b_array,c_array):
    discriminant = b_array**2 - 4*a_array*c_array
    
    return (-b_array-discriminant**0.5) / (2*a_array), (-b_array+discriminant**0.5) / (2*a_array)



'''
Given cross-section lines and the boundary along one side of the ascending aorta, calculate the quadratic fit of the boundary and find the intersection between boundary and cross-section lines. Update boundary with the new intersection points.

'''
def update_quadratic_points(xs, ys, slope, intercept):
    
    coefs = np.polyfit(ys, xs, 2) # [a,b,c] where ax^2 + bx+c
    f_ = np.poly1d(coefs)
    
    # ax^2 + bx + c = mx + k --> ax^2 + (b-m)x + (c-k) solve
    new_coefs = np.ones_like(xs) * coefs[0], np.ones_like(xs) * coefs[1] - slope, np.ones_like(xs) * coefs[2] - intercept
    zero0, zero1 = quadratic_formula(*new_coefs)

    # Choose the intersection point as the zero closer to the original boundary
    dists0, dists1 = np.abs(zero0 - ys), np.abs(zero1 - ys)
    new_ys = (dists0 < dists1) * zero0 + (dists0 >= dists1) * zero1

    new_xs = f_(new_ys)
    
    # Just in case there are no real zeros: Use original boundary 
    new_ys = (1- np.iscomplex(zero0)) * new_ys + np.iscomplex(zero0) * ys
    new_xs = (1- np.iscomplex(zero0)) * new_xs + np.iscomplex(zero0) * xs
    
    return new_xs, new_ys
        
"""
Find orthogonal lines at each point of the skeleton
"""
def xsections_along_skeleton(skeleton, image, pix_size, beta = None, polynomial_fitted = False, remove_root = True, first_point=None, last_point=None):

    skeleton_x, skeleton_y = skeleton[0], skeleton[1]
    
    dists = []
    lines = []
    for i, (ske_x, ske_y) in enumerate(zip(skeleton_x, skeleton_y)):
        if i == 0 or i == len(skeleton_x)-1:
            continue
        tan_x = (skeleton_x[i+1] - skeleton_x[i-1])/2
        tan_y = (skeleton_y[i+1] - skeleton_y[i-1])/2
        mag = (tan_x**2 + tan_y**2)**0.5
        tan_x /= mag
        tan_y /= mag

        x_dir, y_dir = tan_y, -tan_x
        
        # From the medial axis, extend in one direction
        x, y = ske_x, ske_y
        while inside_mask((x,y), image):
            x += x_dir
            y += y_dir
        end_x = x - x_dir
        end_y = y - y_dir

        # extend in the other direction
        x, y = ske_x, ske_y
        while inside_mask((x,y), image):
            x -= x_dir
            y -= y_dir
        start_x = x + x_dir
        start_y = y + y_dir

        # diameter of cross-section
        dist = ((pix_size[0] * (start_x - end_x))**2 + (pix_size[1] * (start_y - end_y))**2)**0.5
        dists.append(dist)
        
        line = [[start_x, end_x], [start_y, end_y]]
        lines.append(line)
        lines_for_drawing = lines
    if polynomial_fitted:
        lines_array = np.array(lines)
        start_xs, start_ys = lines_array[:,0,0], lines_array[:,1,0]
        end_xs, end_ys = lines_array[:,0,1], lines_array[:,1,1]
        
        slope = (end_xs - start_xs) / (end_ys - start_ys)
        intercept = start_xs - slope * start_ys
        

        
        # For each cross-section line, update boundary as the quadratic fitted equations.
        new_start_xs, new_start_ys = update_quadratic_points(start_xs, start_ys, slope, intercept)
        new_end_xs, new_end_ys = update_quadratic_points(end_xs, end_ys, slope, intercept)
        
        
        
        lines = []
        dists = []
        
        for i in range(len(new_start_xs)):
            start_x, end_x, start_y, end_y = new_start_xs[i], new_end_xs[i], new_start_ys[i], new_end_ys[i]
            
            dist = ((pix_size[0] * (start_x - end_x))**2 + (pix_size[1] * (start_y - end_y))**2)**0.5
            dists.append(dist)
        
            line = [[start_x, end_x], [start_y, end_y]]
            lines.append(line)
        dists_before_remove_root = dists
        # Aortic root 부분 더 cut off
        if remove_root:
            
            # diameter와 ascending aorta end point까지 거리 비교하여 diameter가 길면 aortic root or aortic arch로 
            dists_first = (((pix_size[0] * (skeleton_x - first_point[0]))**2 + (pix_size[1] * (skeleton_y - first_point[1]))**2)**0.5)[1:-1]
            dists_last = (((pix_size[0] * (skeleton_x - last_point[0]))**2 + (pix_size[1] * (skeleton_y - last_point[1]))**2)**0.5)[1:-1]
            
            removed = (dists_first > dists) * (dists_last > dists)
            if np.sum(removed) ==0:
                return lines, dists
            lines = list(np.array(lines)[removed])
            lines_for_drawing = list(np.array(lines_for_drawing)[removed])
            dists = list(np.array(dists)[removed])
            
            slope = np.array(slope)[removed]
            intercept = np.array(intercept)[removed]
            skeleton_x = np.array(skeleton_x)[1:-1][removed]
            skeleton_y = np.array(skeleton_y)[1:-1][removed]

            # Check if lines intersect in mask
            axis_symmetry = beta[1] / (-2*beta[0])
            # plt.close()
            # plt.imshow(image.T)
            count = 0
            for i_ in range(len(skeleton_y)):
                if np.abs(skeleton_y[i_] - axis_symmetry) <= 1:
                    continue
                y_ = skeleton_y[i_]
                x_ = skeleton_x[i_]

                axis_y = int(axis_symmetry)
                axis_x = axis_y * slope[i_] + intercept[i_]

                if inside_mask((axis_x, axis_y), image):
                    count += 1
            #         print(axis_y, y_)
            #         plt.plot(x_, y_, "b.")
            #         plt.plot(axis_x, axis_y, "r.")
            # plt.show()
            if count > 10:
                return None, None
    return lines_for_drawing, dists


def preprocess(folder, file_dict, head):
    raw_section = []
    pic_section = []
    zoom_section = []
    for file in file_dict[head]:
        print(file)
        # get image_data
        # img = nibabel.load(os.path.join(folder, file))
        img = nibabel.load(os.path.join(folder, file))
        header_zoom = img.header.get_zooms()
        
        ### Zoom from X-ray
        
#         splitted = file.split("_aorta.png_")
#         xray_file = splitted[0] + "_gt" + splitted[1][0] + ".nii"
#         xray_zoom = nibabel.load(os.path.join("/media/user/data2/aorta_diameter/NLST_4_aorta_proj", xray_file)).header.get_zooms()
        
#         header_zoom = xray_zoom # Use header or xray

        ###
        try:
            image_data = np.squeeze(img.get_fdata())#.T
        except OSError:
            continue

        zoom = (0.7441 * 512 / 2048, 0.7422 * 512 / 2048)
        print(zoom)
        padded_image_data = image_data
        
        """
        if image_data.shape != (STANDARD_DIM, STANDARD_DIM):
            dims = np.where(np.array(image_data.shape) != 1)[0]
            # print(image_data.shape)
            scale_factor = STANDARD_DIM / max(image_data.shape)
            zoom = *(header_zoom[i] / scale_factor for i in dims),
            image_data = rescale(image_data, scale_factor)
            
            padded_image_data = np.ones((STANDARD_DIM, STANDARD_DIM)) * np.min(image_data)
            pad_len = *(int((STANDARD_DIM - image_data.shape[i])/2) for i in dims),
            padded_image_data[pad_len[0]:pad_len[0]+image_data.shape[0], pad_len[1]:pad_len[1]+image_data.shape[1]] = image_data
        else:
            zoom = header_zoom
            # zoom = zoom_from_xray
            padded_image_data = image_data
        """
        
        zoom_section.append(zoom)
        raw_section.append(padded_image_data)
        # pic_section.append((padded_image_data > 0.5).astype(int))
        pic_section.append((padded_image_data > np.percentile(padded_image_data, 50)).astype(int))

    # if i % 100 == 0:
    #     print(i)
    # print(file_list)
    # print("Available NUM in {}".format(list(pic_dict.keys())))
    return pic_section, raw_section, zoom_section
    
# [0, 1] to mapped color with 0.5 as threshold
def combined_cm(value, max_value, map1="Blues", map2="Reds"):
    cm1 = plt.get_cmap(map1)
    cm2 = plt.get_cmap(map2)
    if value < 0.5:
        return cm1(1-2*value)
    else:
        # return cm2(2-2*value)
        return cm2((value-0.5)/(max_value-0.5))
    
def get_largest_mask(image):
    retval, labels, stats, centroids = connectedComponentsWithStats(image)
    largest_mask_i = np.argmax(stats[1:,-1])+1
    largest_mask = (labels == largest_mask_i).astype("uint8")
    
    return largest_mask

def compute_diameter(output_folder, input_folder, file_dict, head, ASCENDING_ONLY=False, ABDOMINAL = False, visualize=True, heat_map = True):
    if ABDOMINAL: ASCENDING_ONLY = True
    
    pic_section, raw_section, zoom_section = preprocess(input_folder, file_dict, head)

    valid_skeleton = True

    ''' Compute for case NUM'''
    start_time = time.time()
    num_files = len(pic_section)

    
    if visualize:
        if not ASCENDING_ONLY:
            fig, axes = plt.subplots(1,num_files + 2, figsize=(4 * (num_files + 2),5))
            
            combined_pic = np.zeros_like(pic_section[0])
            combined_raw = np.zeros_like(pic_section[0])
            
            for num in range(num_files):
                combined_pic = combined_pic + pic_section[num]
                combined_raw = combined_raw + raw_section[num]
        else:
            fig, axes = plt.subplots(1,3, figsize=(12,5))
            combined_pic = pic_section[0]
            combined_raw = raw_section[0]

        axes[0].imshow(combined_raw.T, cmap="gray")
        axes[0].axis("off")
        row_color = ["tab:blue", "tab:orange"]


    combined_dists = []
    annotation = []
    max_dists = []
    max_is = []
    max_line = []
    for row_num in range(num_files):

        pix_size = zoom_section[row_num]
        print("pix_size : ", pix_size)
        image = pic_section[row_num]
        
        
        if row_num == 0: # ascending aorta
            # Smooth edge and separate edge discontinuities.
            kernel = np.ones(shape=(75,75))
            image = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN, kernel)

            largest_mask = get_largest_mask(image)

            scale_factor = 1
            while True:

                
                skeleton_final, first_point, last_point, beta = polynomial_fit(largest_mask, scale_factor)
                if skeleton_final is None:
                    scale_factor *= 2
                else:
                    
                    lines, dists = xsections_along_skeleton(skeleton_final, largest_mask, pix_size, beta=beta, polynomial_fitted=True, remove_root=True, first_point=first_point, last_point=last_point)
                    if lines is not None:
                        break

                    else: scale_factor *= 2
                        
        else: # descending aorta
            kernel = np.ones(shape=(50,50)) # smaller kernel to account for thinner aorta
            image = cv2.morphologyEx(image.astype('uint8'), cv2.MORPH_OPEN, kernel)
            
            largest_mask = get_largest_mask(image)
            
            skeleton_set = skeleton_candidates(largest_mask)
            if len(skeleton_set[0]) == 0:
                valid_skeleton = False
                print('no skeleton')
                return

            skeleton_interpolated = connect_skeleton_candidates(skeleton_set, largest_mask)
            skeleton_final = smooth_gaussian(skeleton_interpolated)
            
            lines, dists = xsections_along_skeleton(skeleton_final, largest_mask, pix_size)
    
        np.save(os.path.join(output_folder, "dists_{}_{}.npy".format(head, row_num)), dists)
        np.save(os.path.join(output_folder, "lines_{}_{}.npy".format(head, row_num)), lines)
        np.save(os.path.join(output_folder, "skel_{}_{}.npy".format(head, row_num)), skeleton_final)

        print("--------------------------------------")
        print("Max diameter: {}mm".format(round(np.nanmax(dists),2)))
        max_line = lines[np.argmax(dists)]
        print("Line point: ({}, {}), ({},{})".format(round(max_line[0][0],2), round(max_line[1][0],2), round(max_line[0][1],2), round(max_line[1][1],2)))
        max_dists.append(max(dists))

        skeleton_x = skeleton_final[0]
        skeleton_y = skeleton_final[1]


        if row_num == 0:
            combined_dists = dists[::-1]
            #annotation = [row_color[row_num] for i in dists]
            #max_is.append(np.argmax(combined_dists))

        else:
            max_is.append(len(combined_dists) + np.argmax(dists))
            combined_dists.extend(dists)
            #annotation.extend([row_color[row_num] for i in dists])

        if visualize:
            axes[row_num+1].imshow(image.T, cmap = "gray")
            axes[row_num+1].axis("off")

            if not heat_map:
                axes[row_num+1].plot(skeleton_x, skeleton_y, row_color[row_num])
                if row_num == 0:
                    axes[row_num+1].plot(first_point[0], first_point[1], color=row_color[row_num], marker="o")
                    axes[row_num+1].plot(last_point[0], last_point[1], color=row_color[row_num], marker="o")

                max_i = np.argmax(dists)
                for i, dist in enumerate(dists):
                    if i == max_i:
                        axes[row_num+1].plot(lines[i][0], lines[i][1], 'r')
                    elif i % 20 == 0:
                        axes[row_num+1].plot(lines[i][0], lines[i][1], 'k', alpha=0.4)
                if row_num == 0:
                    lines_ = np.array(lines)
                    # axes[row_num+1].plot(lines_[:,0,0], lines_[:,1,0], 'g-', linewidth=4)
                    # axes[row_num+1].plot(lines_[:,0,1], lines_[:,1,1], 'g-', linewidth=4)
                    
            else:
                
                
                
                # axes[row_num+1].plot(skeleton_x, skeleton_y, row_color[row_num])

                cm = plt.get_cmap("Reds")
                min_dist, max_dist = np.nanmin(dists), np.nanmax(dists)
                # print(dists)
                # print([min(1, max(0,(dist-40))/(10)) for dist in dists])
                axes[row_num+1].set_prop_cycle('color', [cm(min(1.0, max(0.0,(dist-40))/(10))) for dist in dists])

                for i, dist in enumerate(dists):
                    axes[row_num+1].plot(lines[i][0], lines[i][1], alpha=0.3)
                    
#                 MAP_normal, MAP_aneurysm = 'Blues', 'Reds'
#                 n_steps = 10
#                 threshold = 20 #mm in half segment
#                 for i, dist in enumerate(dists):
#                     for line_dir in range(2):
#                         half_dist = ((pix_size[0]*(skeleton_x[i]-lines[i][0][line_dir]))**2 + 
#                                   (pix_size[1]*(skeleton_y[i]-lines[i][1][line_dir]))**2)**0.5

#                         max_cm = half_dist / (2*threshold) 
                        
#                         # variable n_steps 
#                         if half_dist < 2 * threshold:
#                             xs = np.linspace(skeleton_x[i], lines[i][0][line_dir], n_steps)
#                             ys = np.linspace(skeleton_y[i], lines[i][1][line_dir], n_steps)
#                             new_dists = np.linspace(0, max_cm, n_steps)
#                         else:
#                             xs = np.append(np.linspace(skeleton_x[i], 2*threshold, n_steps), np.linspace(0.5, lines[i][0][line_dir], n_steps))
#                             ys = np.append(np.linspace(skeleton_y[i], 0.5, n_steps), np.linspace(0.5, lines[i][1][line_dir], n_steps))
#                             new_dists = np.append(np.linspace(0, max_cm, n_steps)

#                         axes[row_num+1].set_prop_cycle('color', [combined_cm(new_dists, max_cm, MAP_normal, MAP_aneurysm) for j in range(1,n_steps+1)])
#                         for k in range(n_steps):
#                             axes[row_num+1].plot(xs[k:k+2], ys[k:k+2], alpha=0.3)
            
    if valid_skeleton and visualize:
        x = np.arange(len(combined_dists))
        y = combined_dists
        lines = [((x0,y0), (x1,y1)) for x0, y0, x1, y1 in zip(x[:-1], y[:-1], x[1:], y[1:])]
        colored_lines = LineCollection(lines, colors = annotation)
        lineplot_i = 1 + num_files
        
        axes[lineplot_i].add_collection(colored_lines)
        for num in range(num_files):
            axes[lineplot_i].vlines(max_is[num], ymin=0, ymax=combined_dists[max_is[num]], colors = 'r')

        axes[lineplot_i].set_ylim(np.nanmin(combined_dists), np.nanmax(combined_dists))
        axes[lineplot_i].set_ylabel("Diameter length (mm)")
        # if row_num == 0:
        #     axes[row_num, 2].set_xlabel("Arch -> Ascending -> Root")
        # elif row_num == 1:
        #     axes[row_num, 2].set_xlabel("Arch -> Descending -> Abdomen")
        axes[lineplot_i].set_xlabel("Ascending -> Descending")
        axes[lineplot_i].set_xticks(ticks=[])
        axes[lineplot_i].set_box_aspect(1)
        axes[lineplot_i].set_title("Maximum diameter: {} mm".format(round(np.nanmax(combined_dists), 2)))

        fig.tight_layout()

        plt.savefig(os.path.join(output_folder, "{}.png".format(head)))

    end_time = time.time()
    # print("elapsed time: {} seconds".format(round(end_time - start_time, 2)))
    print("--------------------------------------")
    plt.close()
    
    # Return the max diameters for use in CSV
    return max_dists
