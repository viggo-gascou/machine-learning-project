import os
print("\nImporting libraries (may take some time) ...\n")
from scripts import train_custom_dt, train_sklearn_dt, train_custom_nn

# Gets current working directory
path = os.getcwd()

# Joins the folder that we wanted to create
path = os.path.join(path, "results") 

def guided_training(model, description, folder_name):

    while True:
        answer = input(f"Do you want to train and predict with the {description}? (y/n) ")
        if answer == 'y' or answer == 'Y':
            os.makedirs(os.path.join(path, folder_name), exist_ok=True)
            model() 
            print(f"\nSuccessfully run!\n")
            break
        elif answer == 'n':
            print(f"Didn't train {description}\n")
            break
        else:
            print("Invalid input. Try again.")

def main():

    guided_training(train_custom_nn, "custom feed forward neural network classifier", "custom_NN")

    guided_training(train_custom_dt, "custom decision tree classifier", "custom_DT")

    guided_training(train_sklearn_dt, "scikit-learn decision tree classifier", "sklearn_DT")

    print("Training Pipeline Done!")



if __name__ == "__main__":
    main()
