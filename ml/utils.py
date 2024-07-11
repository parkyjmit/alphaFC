import math


# if the algorithm suggested total time is less than 300s, repeat the proposal integer times until the total time is
# greater than 300s. This is to avoid too frequent EIS step, which is time-consuming
def get_step_actual_time(action, converted=False):
    t_low_bound = 300
    repeat = 1
    print('inside', action)
    t_rest = action[2]
    t_work = action[3]
    t_loop_total = t_rest + t_work if converted else 10**t_rest + 10**t_work
    print('t_loop_total:', t_loop_total)
    # initial step
    if int(t_loop_total) == 0:
        return 0, 0

    elif t_loop_total < t_low_bound:
        # roof the quotient
        repeat = math.ceil(t_low_bound / t_loop_total)

    t_actual = t_loop_total * repeat

    print(f'actual time is {t_actual}\n'
          f'repeat {repeat} times')

    return t_actual, repeat
