import torch

# lagrange interpolation
def lagrange_preint_o1(t1, v1, int_t_start, int_t_end):
    '''
    lagrange interpolation of order 1
    Args:
        t1: timestepx
        v1: value field at t1
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    int1 = (int_t_end-int_t_start)
    return int1*v1, (int1/int1, )

def lagrange_preint_o2(t1, t2, v1, v2, int_t_start, int_t_end):
    '''
    lagrange interpolation of order 2
    Args:
        t1: timestepx
        t2: timestepy
        v1: value field at t1
        v2: value field at t2
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    int1 =  0.5/(t1-t2)*((int_t_end-t2)**2 - (int_t_start-t2)**2)
    int2 =  0.5/(t2-t1)*((int_t_end-t1)**2 - (int_t_start-t1)**2)
    int_sum = int1+int2
    return int1*v1 + int2*v2, (int1/int_sum, int2/int_sum)

def lagrange_preint_o3(t1, t2, t3, v1, v2, v3, int_t_start, int_t_end):
    '''
    lagrange interpolation of order 3
    Args:
        t1: timestepx
        t2: timestepy
        t3: timestepz
        v1: value field at t1
        v2: value field at t2
        v3: value field at t3
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    int1_denom = (t1-t2)*(t1-t3)
    int1_end = 1/3*(int_t_end)**3 - 1/2*(t2+t3)*(int_t_end)**2 + (t2*t3)*int_t_end
    int1_start = 1/3*(int_t_start)**3 - 1/2*(t2+t3)*(int_t_start)**2 + (t2*t3)*int_t_start
    int1 = (int1_end - int1_start)/int1_denom
    int2_denom = (t2-t1)*(t2-t3)
    int2_end = 1/3*(int_t_end)**3 - 1/2*(t1+t3)*(int_t_end)**2 + (t1*t3)*int_t_end
    int2_start = 1/3*(int_t_start)**3 - 1/2*(t1+t3)*(int_t_start)**2 + (t1*t3)*int_t_start
    int2 = (int2_end - int2_start)/int2_denom
    int3_denom = (t3-t1)*(t3-t2)
    int3_end = 1/3*(int_t_end)**3 - 1/2*(t1+t2)*(int_t_end)**2 + (t1*t2)*int_t_end
    int3_start = 1/3*(int_t_start)**3 - 1/2*(t1+t2)*(int_t_start)**2 + (t1*t2)*int_t_start
    int3 = (int3_end - int3_start)/int3_denom
    int_sum = int1+int2+int3
    return int1*v1 + int2*v2 + int3*v3, (int1/int_sum, int2/int_sum, int3/int_sum)

def larange_preint_o4(t1, t2, t3, t4, v1, v2, v3, v4, int_t_start, int_t_end):
    '''
    lagrange interpolation of order 4
    Args:
        t1: timestepx
        t2: timestepy
        t3: timestepz
        t4: timestepw
        v1: value field at t1
        v2: value field at t2
        v3: value field at t3
        v4: value field at t4
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    int1_denom = (t1-t2)*(t1-t3)*(t1-t4)
    int1_end = 1/4*(int_t_end)**4 - 1/3*(t2+t3+t4)*(int_t_end)**3 + 1/2*(t3*t4 + t2*t3 + t2*t4)*int_t_end**2 - t2*t3*t4*int_t_end
    int1_start = 1/4*(int_t_start)**4 - 1/3*(t2+t3+t4)*(int_t_start)**3 + 1/2*(t3*t4 + t2*t3 + t2*t4)*int_t_start**2 - t2*t3*t4*int_t_start
    int1 = (int1_end - int1_start)/int1_denom
    int2_denom = (t2-t1)*(t2-t3)*(t2-t4)
    int2_end = 1/4*(int_t_end)**4 - 1/3*(t1+t3+t4)*(int_t_end)**3 + 1/2*(t3*t4 + t1*t3 + t1*t4)*int_t_end**2 - t1*t3*t4*int_t_end
    int2_start = 1/4*(int_t_start)**4 - 1/3*(t1+t3+t4)*(int_t_start)**3 + 1/2*(t3*t4 + t1*t3 + t1*t4)*int_t_start**2 - t1*t3*t4*int_t_start
    int2 = (int2_end - int2_start)/int2_denom
    int3_denom = (t3-t1)*(t3-t2)*(t3-t4)
    int3_end = 1/4*(int_t_end)**4 - 1/3*(t1+t2+t4)*(int_t_end)**3 + 1/2*(t4*t2 + t1*t2 + t1*t4)*int_t_end**2 - t1*t2*t4*int_t_end
    int3_start = 1/4*(int_t_start)**4 - 1/3*(t1+t2+t4)*(int_t_start)**3 + 1/2*(t4*t2 + t1*t2 + t1*t4)*int_t_start**2 - t1*t2*t4*int_t_start
    int3 = (int3_end - int3_start)/int3_denom
    int4_denom = (t4-t1)*(t4-t2)*(t4-t3)
    int4_end = 1/4*(int_t_end)**4 - 1/3*(t1+t2+t3)*(int_t_end)**3 + 1/2*(t3*t2 + t1*t2 + t1*t3)*int_t_end**2 - t1*t2*t3*int_t_end
    int4_start = 1/4*(int_t_start)**4 - 1/3*(t1+t2+t3)*(int_t_start)**3 + 1/2*(t3*t2 + t1*t2 + t1*t3)*int_t_start**2 - t1*t2*t3*int_t_start
    int4 = (int4_end - int4_start)/int4_denom
    int_sum = int1+int2+int3+int4
    return int1*v1 + int2*v2 + int3*v3 + int4*v4, (int1/int_sum, int2/int_sum, int3/int_sum, int4/int_sum)


def lagrange_preint(order, pre_vs, pre_ts, int_t_start, int_t_end):
    '''
    lagrange interpolation
    Args:
        order: order of interpolation
        pre_vs: value field at pre_ts
        pre_ts: timesteps
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    order = min(order, len(pre_vs), len(pre_ts))
    if order == 1:
        return lagrange_preint_o1(pre_ts[-1], pre_vs[-1], int_t_start, int_t_end)
    elif order == 2:
        return lagrange_preint_o2(pre_ts[-2], pre_ts[-1], pre_vs[-2], pre_vs[-1], int_t_start, int_t_end)
    elif order == 3:
        return lagrange_preint_o3(pre_ts[-3], pre_ts[-2], pre_ts[-1], pre_vs[-3], pre_vs[-2], pre_vs[-1], int_t_start, int_t_end)
    elif order == 4:
        return larange_preint_o4(pre_ts[-4], pre_ts[-3], pre_ts[-2], pre_ts[-1], pre_vs[-4], pre_vs[-3], pre_vs[-2], pre_vs[-1], int_t_start, int_t_end)
    else:
        raise ValueError('Invalid order')


def polynomial_integral(coeffs, int_t_start, int_t_end):
    '''
    polynomial integral
    Args:
        coeffs: coefficients of the polynomial
        int_t_start: intergation start time
        int_t_end: intergation end time
    Returns:
        integrated value
    '''
    orders = len(coeffs)
    int_val = 0
    for o in range(orders):
        int_val += coeffs[o]/(o+1)*(int_t_end**(o+1)-int_t_start**(o+1))
    return int_val

