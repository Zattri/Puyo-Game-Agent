import experience_replay as ExpRep
import training_loop as TrainLoop

def main():
    exp_rep = ExpRep.ExperienceReplay()

    data = exp_rep.readFile("testing")


    for episode in data:
        action1 = TrainLoop.parseIntToString(episode[1][0])
        action2 = TrainLoop.parseIntToString(episode[1][1])
        action3 = TrainLoop.parseIntToString(episode[1][2])
        action4 = TrainLoop.parseIntToString(episode[1][3])
        print(f"{action1} -> {action2} -> {action3} -> {action4}\n")

main()