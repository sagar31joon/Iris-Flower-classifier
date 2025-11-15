# main.py
import os

def main():
    print("\n🌸 IRIS FLOWER CLASSIFICATION SYSTEM 🌸")
    print("---------------------------------------")
    print("Choose a model for prediction : \n")
    print("1. Logistic Regression")
    print("2. Decision Tree Classifier")
    print("3. Support Vector Classifier")
    print("4. Exit")

    while True:
        choice = input("\nEnter your choice (1/2/3/4): ").strip()

        if choice == '1':
            print("\nRunning Logistic Regression model...\n")
            os.system('python3 src/predict_LR.py')
        elif choice == '2':
            print("\nRunning Decision Tree Classifier model...\n")
            os.system('python3 src/predict_decision_tree.py')
        elif choice == '3':
            print("\nRunning Support Vector Classifier model...\n")
            os.system('python3 src/predict_SVM.py')
        elif choice == '4':
            print("\nExiting program. See ya👋!\n")
            break
        else:
            print("Invalid choice! Please enter 1, 2, 3, or 4.")

if __name__ == "__main__":
    main()