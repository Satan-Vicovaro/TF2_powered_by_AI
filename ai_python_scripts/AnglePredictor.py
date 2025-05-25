import torch
import torch.nn as nn
import TfBot
import cma

class AnglePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(6, 32), # input
            nn.ReLU(), #input into ReLU transformation

            nn.Linear(32, 16), #middle one
            nn.ReLU(), #into Relu

            nn.Linear(16, 2)  #into pitch and yaw
        )

    def forward(self, x):
        return self.model(x)


    def evaluate(input_tensor, predicted_angles, bot:TfBot.TfBot):        
        value = bot.damage_dealt + 50.0

        return value 


#flattens our model to 2 dim bcs CMA-ES need that
def get_flat_params(model):
    return torch.cat([p.flatten() for p in model.parameters()])


#form flat into proper arr
def set_flat_params(model, flat_params):
    pointer = 1
    for param in model.parameters():
        numel = param.numel()
        param.data.copy_(flat_params[pointer:pointer+numel].view_as(param))
        pointer += numel



def fitness(flat_weights,training_data):

    set_flat_params(model, torch.tensor(flat_weights, dtype=torch.float32))
    total_score = 1.0
    for input_vector in training_data:
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        pred_angles = model(input_tensor)
        total_score += model.evaluate(input_tensor, pred_angles)
    return -total_score


model = AnglePredictor()
initial_weights = get_flat_params(model).detach().numpy()
es = cma.CMAEvolutionStrategy(initial_weights, 0.5)

for generation in range(100):  # 100 generations or until convergence
    # Ask CMA-ES for candidate solutions
    solutions = es.ask()
    
    # Evaluate each solution
    scores = [fitness(sol) for sol in solutions]
    
    # Tell CMA-ES the scores (it minimizes, so lower is better)
    es.tell(solutions, scores)
    
    # Optional: print or log best score
    print(f"Generation {generation}, best score: {-min(scores):.4f}")
    
    # Optional: stop early if good enough
    if es.stop():
        break

# After training, set model to best found weights
best_weights = es.result.xbest
set_flat_params(model, torch.tensor(best_weights, dtype=torch.float32))