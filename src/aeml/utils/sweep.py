import wandb

wandb.login()

def start_sweep(sweepConfig: dict,
                project_name: str,
                objective: callable,
):
    sweep_id = wandb.sweep(sweepConfig, project=project_name)
    
    def main():
        wandb.init(project=project_name)
        score = objective(wandb.config)
        wandb.log({
            "score": score,
            "skip": sweepConfig['skip']
        })

    wandb.agent(sweep_id, function=main, count = 100)