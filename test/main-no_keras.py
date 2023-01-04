import os
print("\nImporting libraries (may take some time) ...\n")
from scripts import train_custom_dt, train_sklearn_dt, train_custom_nn

# Gets current working directory
path = os.getcwd()

# Joins the folder that we wanted to create
folder_names = ['custom_DT', 'sklearn_DT', 'custom_NN', 'keras_FFNN', 'keras_CNN']
path = os.path.join(path, 'results') 

# Create the results folder 
for folder in folder_names:
    os.makedirs(os.path.join(path, folder), exist_ok=True)

def guided_training(model, description):

    while True:
        answer = input(f'Do you want to train and predict with the {description}? (y/n) ')
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

    guided_training(train_custom_nn, "custom feed forward neural network classifier")

    guided_training(train_custom_dt, "custom decision tree classifier")

    guided_training(train_sklearn_dt, "scikit-learn decision tree classifier")

    print("Training Pipeline Done!")



if __name__ == "__main__":
    main()
