import math
import torch


def find_event(interp_fn, t0, t1, event_fn, tol):
    with torch.no_grad():

        y0 = interp_fn(t0)
        y1 = interp_fn(t1)

        sign0 = torch.sign(event_fn(t0, y0))
        sign1 = torch.sign(event_fn(t1, y1))

        assert sign0 + sign1 == 0, "Signs for ends of interval are the same."

        # Num iterations for the secant method until tolerance is within target.
        nitrs = torch.ceil(torch.log((t1 - t0) / tol) / math.log(2.0))

        for _ in range(nitrs.long()):
            t_mid = (t1 + t0) / 2.0
            y_mid = interp_fn(t_mid)
            sign_mid = torch.sign(event_fn(t_mid, y_mid))
            same_as_sign0 = (sign0 * sign_mid) / 2.0 + 0.5
            t0 = t_mid * same_as_sign0 + t0 * (1.0 - same_as_sign0)
            t1 = t_mid * (1.0 - same_as_sign0) + t1 * same_as_sign0
        event_t = (t0 + t1) / 2.0

    return event_t, interp_fn(event_t)
