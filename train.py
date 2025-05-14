def train(model, x, data, optimizer, critereon):
    model.train()
    
    return train_full_batch(model, x, data, optimizer, critereon )

def train_full_batch(model, x, data, optimizer, critereon):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x, data.edge_index)[data.train_mask] 
    y_true = data.y[data.train_mask].squeeze()

    loss = critereon(y_pred, y_true)
 
    loss.backward()
    optimizer.step()

    return loss
