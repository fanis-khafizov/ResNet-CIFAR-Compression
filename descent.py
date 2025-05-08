import torch
from torch.func import functional_call

def mirror_descent(model, X_train, y_train, lr, eta, lambda_value, num_steps, criterion, errors=None):
    # 1) считаем один раз начальный градиент по весам
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    base_grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    if errors is not None:
        base_grads = [g + errors[name] for g, (name, _) in zip(base_grads, model.named_parameters())]

    # 2) инициализируем impacts
    impacts = {name: torch.zeros_like(param, requires_grad=True) for name, param in model.named_parameters()}

    # 3) подготовим «новые» параметры на каждый шаг
    new_params = {n: p.clone() for n, p in model.named_parameters()}

    # список тех impact‑тензоров, для которых мы считаем градиент
    skip = lambda n: 'bn' in n or 'shortcut.1' in n
    impact_keys = [n for n in impacts if not skip(n)]

    for _ in range(num_steps):
        # a) обновляем new_params по старым весам и base_grads
        for (name, orig_p), g in zip(model.named_parameters(), base_grads):
            if skip(name): continue
            new_params[name] = orig_p - lr * impacts[name] * g

        # b) прямой проход с new_params
        out_new = functional_call(model, new_params, (X_train,))
        loss_new = criterion(out_new, y_train)

        # c) сразу получаем все градиенты impact‑ов
        grads_impacts = torch.autograd.grad(
            loss_new,
            [impacts[k] for k in impact_keys],
            allow_unused=True
        )

        # d) обновляем impacts «вдоль градиента»
        with torch.no_grad():
            for k, g_imp in zip(impact_keys, grads_impacts):
                imp = impacts[k]
                impact_update = torch.pow(imp, 1/(1+eta*lambda_value)).detach() * torch.exp(-(eta/(1+eta*lambda_value)) * (g_imp if g_imp is not None else 0))
                impacts[k] = (impact_update / impact_update.sum() * imp.numel()).detach().requires_grad_(True) 

    # 4) финально отключаем grad, если нужно
    for k in impact_keys:
        impacts[k] = impacts[k].detach().requires_grad_(False)

    return impacts

def gradient_descent(model, X_train, y_train, lr, eta, scale, num_steps, criterion, errors=None):
    # 1) считаем один раз начальный градиент по весам
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    base_grads = torch.autograd.grad(loss, [p for _, p in model.named_parameters()])
    if errors is not None:
        base_grads = [g + errors[name] for g, (name, _) in zip(base_grads, model.named_parameters())]

    # 2) инициализируем impacts
    impacts = {name: torch.zeros_like(param, requires_grad=True) for name, param in model.named_parameters()}

    # 3) подготовим «новые» параметры на каждый шаг
    new_params = {n: p.clone() for n, p in model.named_parameters()}

    # список тех impact‑тензоров, для которых мы считаем градиент
    skip = lambda n: 'bn' in n or 'shortcut.1' in n
    impact_keys = [n for n in impacts if not skip(n)]
    
    for _ in range(num_steps):
        # a) обновляем new_params по старым весам и base_grads
        for (name, orig_p), g in zip(model.named_parameters(), base_grads):
            if skip(name): continue
            new_params[name] = orig_p - lr * impacts[name] * g

        # b) прямой проход с new_params
        out_new = functional_call(model, new_params, (X_train,))
        loss_new = criterion(out_new, y_train)

        # c) сразу получаем все градиенты impact‑ов
        grads_impacts = torch.autograd.grad(
            loss_new,
            [impacts[k] for k in impact_keys],
            allow_unused=True
        )

        # d) обновляем impacts «вдоль градиента»
        with torch.no_grad():
            for k, g_imp in zip(impact_keys, grads_impacts):
                imp = impacts[k]
                upd = imp - eta * lr * (g_imp if g_imp is not None else 0)
                impacts[k] = upd.clamp(0, scale).detach().requires_grad_(True)

    # 4) финально отключаем grad, если нужно
    for k in impact_keys:
        impacts[k] = impacts[k].detach().requires_grad_(False)

    return impacts