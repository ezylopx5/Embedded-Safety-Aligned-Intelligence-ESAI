"""
MATHEMATICAL VERIFICATION OF ESAI-v3 SYSTEM
Traces through all equations to verify correctness.
"""

import numpy as np

print('='*70)
print('MATHEMATICAL ANALYSIS OF ESAI-v3')
print('='*70)

# Config values
gamma_E = 0.9
harm_scale = 0.5
victim_distress = 3.0
tau = 0.1  # temperature
lambda_reg = 10.0
iae_dim = 32

# Expected delta_E for STEAL vs HELP
delta_steal = victim_distress * harm_scale  # 3.0 * 0.5 = 1.5 per dim
delta_help = 0.0

# Vector norms
norm_delta_steal = delta_steal * np.sqrt(iae_dim)  # 1.5 * sqrt(32) = 8.49
norm_delta_help = 0.0

print(f'\n[1] IAE DYNAMICS TARGETS:')
print(f'  STEAL: delta_E = [{delta_steal}] * {iae_dim} dims')
print(f'         ||delta_E|| = {delta_steal} * sqrt({iae_dim}) = {norm_delta_steal:.2f}')
print(f'  HELP:  delta_E = [{delta_help}] * {iae_dim} dims')
print(f'         ||delta_E|| = {norm_delta_help:.2f}')

# Simulate E evolution during episode (agent steals every step)
print(f'\n[2] IAE EVOLUTION (if agent STEALs every step):')
E_norm = 0.0
for t in range(10):
    E_new = gamma_E * E_norm + norm_delta_steal
    E_clipped = min(E_new, 10.0)  # Clipping
    print(f'  Step {t}: ||E|| = {gamma_E:.1f} * {E_norm:.2f} + {norm_delta_steal:.2f} = {E_new:.2f} -> clip -> {E_clipped:.2f}')
    E_norm = E_clipped
    if E_new <= 10.0 and t > 0:
        pass
    if E_clipped == 10.0:
        print(f'  -> E saturates at clip limit 10.0')
        break

# Forecaster predictions (what we expect if trained correctly)
print(f'\n[3] FORECASTER PREDICTIONS (from E_current with ||E||=4):')
E_current_norm = 4.0
E_pred_steal = gamma_E * E_current_norm + norm_delta_steal
E_pred_help = gamma_E * E_current_norm + norm_delta_help
print(f'  E^(STEAL) = {gamma_E} * {E_current_norm} + {norm_delta_steal:.2f} = {E_pred_steal:.2f}')
print(f'  E^(HELP)  = {gamma_E} * {E_current_norm} + {norm_delta_help:.2f} = {E_pred_help:.2f}')

# CRITICAL CHECK: If forecaster has Tanh output (BUG!)
print(f'\n[3b] IF FORECASTER HAS TANH (BUG-035):')
tanh_max = np.sqrt(iae_dim)  # max norm with Tanh is sqrt(32) ≈ 5.66
E_pred_steal_tanh = min(E_pred_steal, tanh_max)
E_pred_help_tanh = min(E_pred_help, tanh_max)
print(f'  E^(STEAL) with Tanh: min({E_pred_steal:.2f}, {tanh_max:.2f}) = {E_pred_steal_tanh:.2f}')
print(f'  E^(HELP) with Tanh:  min({E_pred_help:.2f}, {tanh_max:.2f}) = {E_pred_help_tanh:.2f}')
print(f'  -> Tanh KILLS the difference! Both saturate near {tanh_max:.2f}')

# Harm scalarization R(a) = ||E^(a)||^2
R_steal = E_pred_steal ** 2
R_help = E_pred_help ** 2
print(f'\n[4] HARM SCALARIZATION R(a) = ||E^(a)||²:')
print(f'  R(STEAL) = {E_pred_steal:.2f}² = {R_steal:.2f}')
print(f'  R(HELP)  = {E_pred_help:.2f}² = {R_help:.2f}')

# Softmin weights
print(f'\n[5] SOFTMIN REFERENCE (τ = {tau}):')
log_w_steal = -R_steal / tau
log_w_help = -R_help / tau
print(f'  log π_ref(STEAL) = -{R_steal:.2f}/{tau} = {log_w_steal:.2f}')
print(f'  log π_ref(HELP)  = -{R_help:.2f}/{tau} = {log_w_help:.2f}')

# Softmax normalization
log_max = max(log_w_steal, log_w_help)
w_steal = np.exp(log_w_steal - log_max)
w_help = np.exp(log_w_help - log_max)
total = w_steal + w_help
w_steal /= total
w_help /= total
print(f'  π_ref(STEAL) ≈ {w_steal:.10f}')
print(f'  π_ref(HELP)  ≈ {w_help:.10f}')

# Reference embedding
E_ref_norm = w_steal * E_pred_steal + w_help * E_pred_help
print(f'  ||E^ref|| = {w_steal:.4f}*{E_pred_steal:.2f} + {w_help:.4f}*{E_pred_help:.2f} = {E_ref_norm:.2f}')

# Alignment Regret
print(f'\n[6] ALIGNMENT REGRET AR(a) = ||E^(a) - E^ref||²:')
AR_steal = (E_pred_steal - E_ref_norm) ** 2
AR_help = (E_pred_help - E_ref_norm) ** 2
print(f'  AR(STEAL) = ({E_pred_steal:.2f} - {E_ref_norm:.2f})² = {AR_steal:.2f}')
print(f'  AR(HELP)  = ({E_pred_help:.2f} - {E_ref_norm:.2f})² = {AR_help:.2f}')

# Reward transformation
print(f'\n[7] REWARD TRANSFORMATION r\'= r - λ*AR (λ = {lambda_reg}):')
r_steal = 5.0
r_help = 1.0
r_prime_steal = r_steal - lambda_reg * AR_steal
r_prime_help = r_help - lambda_reg * AR_help
print(f'  r\'(STEAL) = {r_steal} - {lambda_reg}*{AR_steal:.2f} = {r_prime_steal:.2f}')
print(f'  r\'(HELP)  = {r_help} - {lambda_reg}*{AR_help:.2f} = {r_prime_help:.2f}')
print(f'\n  DIFFERENCE: r\'(HELP) - r\'(STEAL) = {r_prime_help - r_prime_steal:.2f}')

if r_prime_help > r_prime_steal:
    print(f'  ✓ HELP is preferred by {r_prime_help - r_prime_steal:.2f} reward units')
else:
    print(f'  ✗ STEAL is STILL preferred! BUG EXISTS.')

# Now show what happens with Tanh bug
print('\n' + '='*70)
print('WHAT HAPPENS WITH TANH BUG (BUG-035):')
print('='*70)

R_steal_tanh = E_pred_steal_tanh ** 2
R_help_tanh = E_pred_help_tanh ** 2
log_w_steal_t = -R_steal_tanh / tau
log_w_help_t = -R_help_tanh / tau
log_max_t = max(log_w_steal_t, log_w_help_t)
w_steal_t = np.exp(log_w_steal_t - log_max_t)
w_help_t = np.exp(log_w_help_t - log_max_t)
total_t = w_steal_t + w_help_t
w_steal_t /= total_t
w_help_t /= total_t

E_ref_tanh = w_steal_t * E_pred_steal_tanh + w_help_t * E_pred_help_tanh
AR_steal_tanh = (E_pred_steal_tanh - E_ref_tanh) ** 2
AR_help_tanh = (E_pred_help_tanh - E_ref_tanh) ** 2

r_prime_steal_t = r_steal - lambda_reg * AR_steal_tanh
r_prime_help_t = r_help - lambda_reg * AR_help_tanh

print(f'  R(STEAL) = {R_steal_tanh:.2f}, R(HELP) = {R_help_tanh:.2f}')
print(f'  AR(STEAL) = {AR_steal_tanh:.2f}, AR(HELP) = {AR_help_tanh:.2f}')
print(f'  r\'(STEAL) = {r_prime_steal_t:.2f}, r\'(HELP) = {r_prime_help_t:.2f}')
print(f'  DIFFERENCE with Tanh: {r_prime_help_t - r_prime_steal_t:.2f}')
print(f'\n  -> With Tanh: difference is {abs(r_prime_help_t - r_prime_steal_t):.2f}')
print(f'  -> Without Tanh: difference is {abs(r_prime_help - r_prime_steal):.2f}')
print(f'  -> Tanh reduces discriminative power by {abs(r_prime_help - r_prime_steal) / max(abs(r_prime_help_t - r_prime_steal_t), 0.01):.1f}x!')

print('\n' + '='*70)
print('CONCLUSION: Tanh on forecaster output is CRITICAL BUG')
print('Fix: Remove Tanh from forecast_net in model.py')
print('='*70)
