import os
from scripts import train_custom_nn, train_custom_dt, train_keras_cnn

# Gets current working directory
path = os.getcwd()

# Joins the folder that we wanted to create
folder_name = 'results'
path = os.path.join(path, folder_name) 

# Create the results folder 
os.makedirs(path, exist_ok=True)

def guided_training(model, description):

    while True:
        answer = input(f'Do you want to train and predict the {description}? (y/n) ')
        if answer == 'y' or answer == 'Y':
            model() 
            print(f"\nSuccessfully run!\n")
            break
        elif answer == 'n':
            print(f"Didn't train {description}\n")
            break
        else:
            print("Invalid input. Try again.")

def main():
    guided_training(train_custom_nn, "custom feed foward neural network classifier")

    guided_training(train_custom_dt, "custom decision tree classifier")

    guided_training(train_keras_cnn, "keras convolutional neural network classifer")

    print("Training Pipeline Done!")



if __name__ == "__main__":
    main()
