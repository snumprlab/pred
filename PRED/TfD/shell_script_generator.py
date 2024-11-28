def generate_commands(num_gpu, unseen_episodes, seen_episodes, num_code_per_gpu=6):
    commands = []
    seen_ep = 0
    unseen_ep = 0
    seen_gap_per_gpu = seen_episodes // num_gpu
    unseen_gap_per_gpu = unseen_episodes // num_gpu
    seen_gap = seen_gap_per_gpu // num_code_per_gpu
    unseen_gap = unseen_gap_per_gpu // num_code_per_gpu
    
    for gpu in range(num_gpu):
        for i in range(num_code_per_gpu):
            if gpu  == num_gpu - 1 and i == num_code_per_gpu - 1:
                command = (
                    f"(bash validation_seen_tfd.sh {gpu} {seen_ep} {seen_episodes_total} && "
                    f"bash validation_unseen_tfd.sh {gpu} {unseen_ep} {unseen_episodes_total})&"
                )
            elif i == num_code_per_gpu - 1:
                command = (
                    f"(bash validation_seen_tfd.sh {gpu} {seen_ep} {seen_gap_per_gpu*(gpu+1)} && "
                    f"bash validation_unseen_tfd.sh {gpu} {unseen_ep} {unseen_gap_per_gpu*(gpu+1)})&"
                )
                seen_ep = seen_gap_per_gpu*(gpu+1)
                unseen_ep = unseen_gap_per_gpu*(gpu+1)
            else:
                command = (
                    f"(bash validation_seen_tfd.sh {gpu} {seen_ep} {seen_ep + seen_gap} && "
                    f"bash validation_unseen_tfd.sh {gpu} {unseen_ep} {unseen_ep + unseen_gap})&"
                )
                seen_ep += seen_gap
                unseen_ep += unseen_gap
            commands.append(command)

        commands.append("\n")
    return commands


num_gpu = 4
unseen_episodes_total = 269
seen_episodes_total = 77

commands = generate_commands(num_gpu, unseen_episodes_total, seen_episodes_total)

for command in commands:
    print(command)
    
