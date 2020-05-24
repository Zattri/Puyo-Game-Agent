import experience_replay as ExpRep
import matplotlib.pyplot as plt

def main():
    data = ExpRep.readFile("test")

    plt.imshow(data[0][0][0])
    plt.show()

main()