import ast
import matplotlib.pyplot as plt
import numpy as np

def main():
    print("select folder with data form `statistics_and_data`")
    folder_name = input()
    
    with open("statistics_and_data/" + folder_name + "/accuracy", "r") as f:
        content = f.read()
    
    accuracy = ast.literal_eval(content)
    
    with open("statistics_and_data/" + folder_name + "/avg_reward", "r") as f:
        content = f.read()
    
    avg_reward = ast.literal_eval(content)
    
    with open("statistics_and_data/" + folder_name + "/sum_reward", "r") as f:
        content = f.read()
    
    sum_reward = ast.literal_eval(content)

    # Create subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))

    y = [float(x)  for x in accuracy]
    # Compress by averaging every 500 values
    chunk_size = 1000
    compressed = [
        sum(y[i:i+chunk_size]) / chunk_size
        for i in range(0, len(y), chunk_size)
    ]
    # Create the plot
    axs[0].plot( compressed, linestyle='-')

    # Add labels and title
    axs[0].set_xlabel("X-axis")
    axs[0].set_ylabel("Y-axis")
    axs[0].set_title("Accuracy")

    y = [float(x)  for x in avg_reward]
    # Compress by averaging every 500 values
    chunk_size = 100
    compressed = [
        sum(y[i:i+chunk_size]) / chunk_size
        for i in range(0, len(y), chunk_size)
    ]
    # Create the plot
    axs[1].plot(compressed, linestyle='-')

    # Add labels and title
    axs[1].set_xlabel("X-axis")
    axs[1].set_ylabel("Y-axis")
    axs[1].set_title("Avg reward")


    y = [float(x)  for x in sum_reward]
    # Compress by averaging every 500 values
    chunk_size = 100
    compressed = [
        sum(y[i:i+chunk_size]) / chunk_size
        for i in range(0, len(y), chunk_size)
    ]
    # Create the plot
    axs[2].plot( compressed, linestyle='-')

    # Add labels and title
    axs[2].set_xlabel("X-axis")
    axs[2].set_ylabel("Y-axis")
    axs[2].set_title("Reward sum")
    
    # Show the plot
    plt.show()




if __name__ == "__main__":
    main()