import argparse
import os
import sys

def run_data_generation():
    print("=== Step 1: Generating Dataset ===")
    from data.generate_dataset import main as generate
    generate()

def run_training():
    print("\n=== Step 2: Training Classical Models ===")
    from models.train import train_models
    train_models()

def run_dl_training():
    print("\n=== Step 2.5: Training Deep Learning (Bi-LSTM) ===")
    from models.train_dl import train_dl_model
    train_dl_model()

def run_evaluation():
    print("\n=== Step 3: Evaluating Best Model ===")
    from models.evaluate import evaluate_best_model
    evaluate_best_model()

def run_webapp():
    print("\n=== Step 4: Starting Web Application ===")
    # Web app must be run as a module because of relative imports
    os.system("python webapp/app.py")

def main():
    parser = argparse.ArgumentParser(description="Vulgar Language Detection via Video - ML Pipeline")
    parser.add_argument('--all', action='store_true', help="Run the entire pipeline (generate->train->evaluate->webapp)")
    parser.add_argument('--generate', action='store_true', help="Only generate the dataset")
    parser.add_argument('--train', action='store_true', help="Only run classical model training")
    parser.add_argument('--train-dl', action='store_true', help="Train the deep learning model")
    parser.add_argument('--eval', action='store_true', help="Only evaluate models")
    parser.add_argument('--web', action='store_true', help="Only start the web application")
    
    args = parser.parse_args()
    
    if len(sys.argv) == 1:
        # If no arguments provided, show help
        parser.print_help()
        print("\nUsing default behavior: Starting Web App (assuming models exist).")
        print("If this is your first run, please execute: python main.py --all")
        # Run Webapp
        if os.path.exists('outputs/models/best_model.joblib'):
            run_webapp()
        else:
            print("Error: Models not found. You must run with --all or --train first.")
    else:
        if args.all or args.generate:
            run_data_generation()
        if args.all or args.train:
            run_training()
        if args.all or args.train_dl:
            run_dl_training()
        if args.all or args.eval:
            run_evaluation()
        if args.all or args.web:
            run_webapp()

if __name__ == "__main__":
    main()
