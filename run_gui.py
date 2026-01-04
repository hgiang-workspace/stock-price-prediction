import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import threading
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import sys
import os
sys.path.append(os.getcwd())

from src.models.linear_regression import LinearRegression
from src.models.random_forest import RandomForestRegressor
from src.models.arima import ARIMA
from src.models.ridge import RidgeRegression
from src.models.stacking_ensemble import StackingEnsemble
from src.evaluation.metrics import compute_regression_metrics
from src.data_processing.preprocessing import preprocess_and_merge
from src.data_processing.feature_engineering import build_feature_set
from src.data_processing.train_test_split import split_and_scale
from src.data_collection import download_stock_data, download_market_index

class StockPredictionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Stock Price Prediction - Desktop App")
        self.root.geometry("1400x900")
        
        # Data State
        self.df = None
        self.output_path = None

        # Style
        style = ttk.Style()
        style.theme_use('clam')
        
        # Tabs
        self.tabControl = ttk.Notebook(root)
        
        self.tab_home = ttk.Frame(self.tabControl)
        self.tab_prediction = ttk.Frame(self.tabControl)
        self.tab_comparison = ttk.Frame(self.tabControl)
        
        self.tabControl.add(self.tab_home, text='Home')
        self.tabControl.add(self.tab_prediction, text='Prediction')
        self.tabControl.add(self.tab_comparison, text='Comparison')
        
        self.tabControl.pack(expand=1, fill="both")
        
        # SETUP HOME TAB
        self.setup_home_tab()
        self.setup_prediction_tab()
        self.setup_comparison_tab()

    def log(self, message):
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_area.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_area.see(tk.END)

    def setup_home_tab(self):
        self.tab_home.columnconfigure(0, weight=1)
        self.tab_home.columnconfigure(1, weight=3)
        self.tab_home.rowconfigure(0, weight=1)
        
        # LEFT PANEL
        left_frame = ttk.Frame(self.tab_home, padding="10")
        left_frame.grid(row=0, column=0, sticky="nsew")
        
        ttk.Label(left_frame, text="DATA CONFIGURATION", font=("Arial", 12, "bold")).pack(pady=10)
        
        # Ticker
        ttk.Label(left_frame, text="Ticker Symbol:").pack(anchor="w")
        self.ticker_var = tk.StringVar(value="AAPL")
        self.ticker_combo = ttk.Combobox(left_frame, textvariable=self.ticker_var)
        self.ticker_combo['values'] = ("AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA")
        self.ticker_combo.pack(fill="x", pady=5)
        
        # Date Range
        ttk.Label(left_frame, text="Start Date (YYYY-MM-DD):").pack(anchor="w")
        self.start_date_var = tk.StringVar(value="2000-01-01")
        ttk.Entry(left_frame, textvariable=self.start_date_var).pack(fill="x", pady=5)
        
        ttk.Label(left_frame, text="End Date (YYYY-MM-DD):").pack(anchor="w")
        self.end_date_var = tk.StringVar(value="2026-01-01")
        ttk.Entry(left_frame, textvariable=self.end_date_var).pack(fill="x", pady=5)
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=10)
        
        # Market Indices
        ttk.Label(left_frame, text="Merge Market Indices:").pack(anchor="w")
        self.use_vix = tk.BooleanVar(value=True)
        self.use_nasdaq = tk.BooleanVar(value=True)
        self.use_tnx = tk.BooleanVar(value=True)
        
        ttk.Checkbutton(left_frame, text="VIX (^VIX)", variable=self.use_vix).pack(anchor="w")
        ttk.Checkbutton(left_frame, text="NASDAQ (^IXIC)", variable=self.use_nasdaq).pack(anchor="w")
        ttk.Checkbutton(left_frame, text="TNX (^TNX)", variable=self.use_tnx).pack(anchor="w")
        
        ttk.Separator(left_frame, orient='horizontal').pack(fill='x', pady=20)
        
        # Buttons
        self.btn_load = ttk.Button(left_frame, text="1. DOWNLOAD & LOAD DATA", command=self.load_data_thread)
        self.btn_load.pack(fill="x", pady=5, ipady=5)
        
        self.btn_predict = ttk.Button(left_frame, text="2. TRAIN & PREDICT", command=self.predict_thread, state="disabled")
        self.btn_predict.pack(fill="x", pady=5, ipady=5)
        
        # RIGHT PANEL
        right_frame = ttk.Frame(self.tab_home, padding="10")
        right_frame.grid(row=0, column=1, sticky="nsew")
        
        # Log Area
        ttk.Label(right_frame, text="System Logs:").pack(anchor="w")
        self.log_area = scrolledtext.ScrolledText(right_frame, height=8, state='normal')
        self.log_area.pack(fill="x", pady=5)
        
        # Chart Area
        ttk.Label(right_frame, text="Prediction Visualization:").pack(anchor="w", pady=(10, 0))
        
        self.chart_frame = ttk.Frame(right_frame, borderwidth=2, relief="sunken")
        self.chart_frame.pack(expand=True, fill="both")
        
        # Matplotlib Figure
        self.fig = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Stock Price Chart")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price")
        self.ax.grid(True)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill="both", expand=True)

    def setup_prediction_tab(self):
        paned = ttk.PanedWindow(self.tab_prediction, orient=tk.HORIZONTAL)
        paned.pack(fill="both", expand=True, padx=5, pady=5)
        
        # LEFT PANE (Controls)
        left_pane = ttk.Frame(paned, padding="5")
        paned.add(left_pane, weight=1)
        
        ttk.Label(left_pane, text="SELECT MODEL", font=("Arial", 12, "bold")).pack(pady=10)
        
        self.model_var = tk.StringVar(value="Linear Regression")
        models = [
            ("Linear Regression", "src/models/linear_regression.py"),
            ("Random Forest", "src/models/random_forest.py"),
            ("ARIMA", "src/models/arima.py"),
            ("Ridge Regression", "src/models/ridge.py"),
        ]
        
        self.model_paths = {m[0]: m[1] for m in models}
        
        for name, path in models:
            ttk.Radiobutton(left_pane, text=name, variable=self.model_var, value=name, 
                            command=self.display_source_code).pack(anchor="w", pady=2)
                            
        ttk.Separator(left_pane, orient='horizontal').pack(fill='x', pady=20)
        
        self.btn_train_single = ttk.Button(left_pane, text="TRAIN & EVALUATE", command=self.train_single_model_thread)
        self.btn_train_single.pack(fill="x", pady=10, ipady=5)
        
        # Metrics Table
        ttk.Label(left_pane, text="Evaluation Metrics:").pack(anchor="w", pady=(20, 5))
        columns = ("RMSE", "MAE", "DA", "R2")
        self.metrics_tree = ttk.Treeview(left_pane, columns=columns, show="headings", height=2)
        
        for col in columns:
            self.metrics_tree.heading(col, text=col)
            self.metrics_tree.column(col, width=60, anchor="center")
            
        self.metrics_tree.pack(fill="x")

        # RIGHT PANE (Details)
        right_pane = ttk.Frame(paned, padding="5")
        paned.add(right_pane, weight=3)
        
        ttk.Label(right_pane, text="Model Source Code:").pack(anchor="w")
        self.source_code_area = scrolledtext.ScrolledText(right_pane, height=15, font=("Consolas", 10))
        self.source_code_area.pack(fill="x", pady=5)
        
        ttk.Label(right_pane, text="Single Model Prediction:").pack(anchor="w", pady=(10,0))
        self.chart_frame_single = ttk.Frame(right_pane, borderwidth=2, relief="sunken")
        self.chart_frame_single.pack(expand=True, fill="both")
        
        self.fig_single = plt.Figure(figsize=(5, 3), dpi=100)
        self.ax_single = self.fig_single.add_subplot(111)
        self.ax_single.set_title("Single Model Prediction")
        self.ax_single.grid(True)
        
        self.canvas_single = FigureCanvasTkAgg(self.fig_single, master=self.chart_frame_single)
        self.canvas_single.draw()
        self.canvas_single.get_tk_widget().pack(fill="both", expand=True)
        
        # Load initial source code
        self.display_source_code()

    def display_source_code(self):
        model_name = self.model_var.get()
        path = self.model_paths.get(model_name, "")
        
        self.source_code_area.delete(1.0, tk.END)
        
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            self.source_code_area.insert(tk.END, content)
        else:
            self.source_code_area.insert(tk.END, f"File not found: {path}")

    def train_single_model_thread(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data in 'Home' tab first!")
            return
            
        self.btn_train_single.config(state="disabled")
        threading.Thread(target=self.run_single_model, daemon=True).start()

    def run_single_model(self):
        try:
            model_name = self.model_var.get()
            print(f"Training {model_name}...")
            
            df_fe = self.df.copy()
            # Prepare Data same as Stacking
            original_dates = pd.to_datetime(df_fe.index if df_fe.index.name == 'Date' else df_fe['Date'])
            
            df_fe = build_feature_set(df_fe)
            df_fe.dropna(inplace=True)
            print(df_fe.shape)
            stacking = StackingEnsemble()
            df_fe = stacking.prepare_target(df=df_fe)
            print(df_fe.shape)
            
            train_val_df, test_df = split_and_scale(df_fe, train_ratio=0.8, target="y_target")
            
            FEATURE_COLS = ["return", "lag_1", "lag_5", "ma_20", "volatility_10", "VIX", "NASDAQ", "TNX"]
            
            X_train = train_val_df[FEATURE_COLS].values
            y_train = train_val_df["y_target"].values
            X_test = test_df[FEATURE_COLS].values
            y_test = test_df["y_target"].values
            
            y_pred = None
            
            # Train specific model
            if model_name == "Linear Regression":
                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test).flatten()
                
            elif model_name == "Random Forest":
                model = RandomForestRegressor()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_name == "Ridge Regression":
                model = RidgeRegression()
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                
            elif model_name == "ARIMA":
                train_ts = train_val_df["Adj Close"].values
                model = ARIMA(train_ts, p=5, d=1, q=1, steps=len(test_df))
                y_pred = model 
                
            # Evaluate
            metrics = compute_regression_metrics(y_test, y_pred)
            
            # Update UI
            self.root.after(0, lambda: self.update_single_results(metrics, test_df, original_dates, y_test, y_pred))
            
        except Exception as e:
            messagebox.showerror("Experiment Error", str(e))
            self.root.after(0, lambda: self.btn_train_single.config(state="normal"))

    def update_single_results(self, metrics, test_df, original_dates, y_true, y_pred):
        # Update Table
        for item in self.metrics_tree.get_children():
            self.metrics_tree.delete(item)
            
        vals = metrics.iloc[0].tolist()
        formatted_vals = [f"{v:.4f}" for v in vals]
        self.metrics_tree.insert("", "end", values=formatted_vals)
        
        # Update Chart
        test_start_pos = len(original_dates) - len(test_df)
        test_dates = original_dates[-len(test_df):]
        
        self.ax_single.clear()
        self.ax_single.plot(test_dates, y_true, label="Actual", color='blue', alpha=0.6)
        self.ax_single.plot(test_dates, y_pred, label=self.model_var.get(), color='green', alpha=0.8)
        self.ax_single.legend()
        self.ax_single.set_title(f"{self.model_var.get()} Performance")
        self.ax_single.grid(True)
        self.fig_single.tight_layout()
        self.canvas_single.draw()
        
        self.btn_train_single.config(state="normal")
        messagebox.showinfo("Success", f"Trained {self.model_var.get()} successfully!")

    def setup_comparison_tab(self):
        top_frame = ttk.Frame(self.tab_comparison, padding="10")
        top_frame.pack(fill="x")
        
        self.btn_compare = ttk.Button(top_frame, text="RUN FULL COMPARISON (5 Models)", command=self.run_comparison_thread)
        self.btn_compare.pack(fill="x", ipady=10)
        
        middle_frame = ttk.Frame(self.tab_comparison, padding="10")
        middle_frame.pack(fill="x")
        
        ttk.Label(middle_frame, text="Model Comparison Metrics:", font=("Arial", 11, "bold")).pack(anchor="w")
        
        columns = ("Model", "RMSE", "MAE", "DA", "R2")
        self.comp_tree = ttk.Treeview(middle_frame, columns=columns, show="headings", height=6)
        
        self.comp_tree.heading("Model", text="Model")
        self.comp_tree.column("Model", width=150, anchor="w")
        for col in columns[1:]:
            self.comp_tree.heading(col, text=col)
            self.comp_tree.column(col, width=80, anchor="center")
            
        self.comp_tree.pack(fill="x", pady=5)
        
        bottom_frame = ttk.Frame(self.tab_comparison, padding="10")
        bottom_frame.pack(fill="both", expand=True)
        
        self.fig_comp = plt.Figure(figsize=(5, 4), dpi=100)
        self.ax_comp = self.fig_comp.add_subplot(111)
        self.ax_comp.set_title("Model Comparison Chart")
        self.ax_comp.grid(True)
        
        self.canvas_comp = FigureCanvasTkAgg(self.fig_comp, master=bottom_frame)
        self.canvas_comp.draw()
        self.canvas_comp.get_tk_widget().pack(fill="both", expand=True)

    def run_comparison_thread(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data in 'Home' tab first!")
            return
            
        self.btn_compare.config(state="disabled")
        threading.Thread(target=self.run_comparison, daemon=True).start()

    def run_comparison(self):
        try:
            self.log("=== STARTING FULL COMPARISON ===")
            
            # Prepare Data
            df_fe = self.df.copy()
            original_dates = pd.to_datetime(df_fe.index if df_fe.index.name == 'Date' else df_fe['Date'])
            df_fe = build_feature_set(df_fe)
            df_fe.dropna(inplace=True)
            
            stacking_helper = StackingEnsemble()
            df_fe = stacking_helper.prepare_target(df_fe)
            
            train_val_df, test_df = split_and_scale(df_fe, train_ratio=0.8, target="y_target")
            
            FEATURE_COLS = ["return", "lag_1", "lag_5", "ma_20", "volatility_10", "VIX", "NASDAQ", "TNX"]
            
            X_train = train_val_df[FEATURE_COLS].values
            y_train = train_val_df["y_target"].values
            X_test = test_df[FEATURE_COLS].values
            y_test = test_df["y_target"].values
            
            results = {} 
            metrics_list = []
            
            # Linear Regression
            self.log("Training Linear Regression...")
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            pred_lr = lr.predict(X_test).flatten()
            results["Linear Regression"] = pred_lr
            metrics_list.append(("Linear Regression", compute_regression_metrics(y_test, pred_lr)))
            
            # Random Forest
            self.log("Training Random Forest...")
            rf = RandomForestRegressor()
            rf.fit(X_train, y_train)
            pred_rf = rf.predict(X_test).flatten()
            results["Random Forest"] = pred_rf
            metrics_list.append(("Random Forest", compute_regression_metrics(y_test, pred_rf)))
            
            # Ridge
            self.log("Training Ridge Regression...")
            ridge = RidgeRegression()
            ridge.fit(X_train, y_train)
            pred_ridge = ridge.predict(X_test).flatten()
            results["Ridge Regression"] = pred_ridge
            metrics_list.append(("Ridge Regression", compute_regression_metrics(y_test, pred_ridge)))
            
            # ARIMA
            self.log("Training ARIMA...")
            train_ts = train_val_df["Adj Close"].values
            arima = ARIMA(train_ts, p=5, d=1, q=1, steps=len(test_df))
            pred_arima = arima.flatten() 
            results["ARIMA"] = pred_arima
            metrics_list.append(("ARIMA", compute_regression_metrics(y_test, pred_arima)))
            
            # Stacking Ensemble
            self.log("Training Stacking Ensemble (Full Pipeline)...")
            stacking = StackingEnsemble(ridge_alpha=0.001)
            # Re-train proper pipeline for Stacking
            stacking.train_base_models_walk_forward(train_val_df, FEATURE_COLS)
            stacking.train_meta_model()
            stacking.retrain_base_models_full(train_val_df, FEATURE_COLS)
            pred_stacking = stacking.predict_test(test_df, FEATURE_COLS).flatten()
            results["Stacking Ensemble"] = pred_stacking
            metrics_list.append(("Stacking Ensemble", compute_regression_metrics(y_test, pred_stacking)))
            
            # Update UI
            self.root.after(0, lambda: self.update_comparison_ui(metrics_list, results, original_dates, test_df, y_test))
            self.log("=== COMPARISON COMPLETED ===")
            
        except Exception as e:
            self.log(f"COMPARISON ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.btn_compare.config(state="normal"))

    def update_comparison_ui(self, metrics_list, results, original_dates, test_df, y_true):
        # Update Table
        for item in self.comp_tree.get_children():
            self.comp_tree.delete(item)
            
        for name, m_df in metrics_list:
            vals = m_df.iloc[0].tolist()
            row = [name] + [f"{v:.4f}" for v in vals]
            self.comp_tree.insert("", "end", values=row)
            
        # Update Chart
        test_dates = original_dates[-len(test_df):]
        
        self.ax_comp.clear()
        
        # Plot Actual
        self.ax_comp.plot(test_dates, y_true, label="Actual Price", color='black', linewidth=2, linestyle='--')
        
        # Plot Models
        colors = ['red', 'green', 'orange', 'purple', 'blue']
        for i, (name, pred) in enumerate(results.items()):
            color = colors[i % len(colors)]
            self.ax_comp.plot(test_dates, pred, label=name, color=color, alpha=0.7, linewidth=1)
            
        self.ax_comp.set_title("All Models Comparison")
        self.ax_comp.set_xlabel("Date")
        self.ax_comp.set_ylabel("Price")
        self.ax_comp.legend()
        self.ax_comp.grid(True)
        self.fig_comp.tight_layout()
        
        self.canvas_comp.draw()
        
        self.btn_compare.config(state="normal")
        messagebox.showinfo("Success", "Comparison Completed!")

    def load_data_thread(self):
        self.btn_load.config(state="disabled")
        threading.Thread(target=self.run_load_data, daemon=True).start()

    def run_load_data(self):
        try:
            ticker = self.ticker_var.get()
            start = self.start_date_var.get()
            end = self.end_date_var.get()
            
            self.log(f"Starting data download for {ticker} ({start} to {end})...")
            
            # Download Stock
            download_stock_data(ticker, start, end)
            
            # Market Indices
            indices = []
            if self.use_vix.get():
                download_market_index("^VIX", "VIX", start, end)
                indices.append("VIX")
            if self.use_nasdaq.get():
                download_market_index("^IXIC", "NASDAQ", start, end)
                indices.append("NASDAQ")
            if self.use_tnx.get():
                download_market_index("^TNX", "TNX", start, end)
                indices.append("TNX")
            
            self.log("Merging data...")
            df , output_path = preprocess_and_merge(ticker, indices, start, end)
            self.df = df
            
            self.log(f"Data loaded successfully! Shape: {self.df.shape}")
            self.root.after(0, lambda: self.btn_predict.config(state="normal"))
            self.root.after(0, lambda: self.btn_load.config(state="normal"))
            
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            self.root.after(0, lambda: self.btn_load.config(state="normal"))

    def predict_thread(self):
        if self.df is None:
            messagebox.showerror("Error", "Please load data first!")
            return
            
        self.btn_predict.config(state="disabled")
        threading.Thread(target=self.run_prediction, daemon=True).start()

    def run_prediction(self):
        try:
            self.log("Building Feature Set...")
            df_fe = self.df.copy()
            
            # Save original dates for plotting
            original_dates = pd.to_datetime(df_fe.index if df_fe.index.name == 'Date' else df_fe['Date'])
            
            df_fe = build_feature_set(df_fe)
            df_fe.dropna(inplace=True)
            
            # Config
            FEATURE_COLS = ["return", "lag_1", "lag_5", "ma_20", "volatility_10", "VIX", "NASDAQ", "TNX"]
            
            self.log("Initializing Stacking Ensemble...")
            stacking = StackingEnsemble(
                ridge_alpha=0.001,
                start_year=2000,
                first_train_end_year=2018,
                meta_train_year=2019
            )
            
            df_fe = stacking.prepare_target(df_fe)
            train_val_df, test_df = split_and_scale(df_fe, train_ratio=0.8, target="y_target")
            
            self.log("Training Base Models (Walk-forward)...")
            stacking.train_base_models_walk_forward(train_val_df, FEATURE_COLS)
            
            self.log("Training Meta Model...")
            stacking.train_meta_model()
            
            self.log("Retraining Base Models on full train set...")
            stacking.retrain_base_models_full(train_val_df, FEATURE_COLS)
            
            self.log("Predicting on Test Set...")
            y_pred = stacking.predict_test(test_df, FEATURE_COLS)
            y_true = test_df["y_target"].values
            
            metrics = stacking.evaluate(y_true, y_pred)
            
            # Format metrics string
            metrics_str = ", ".join([f"{col}: {val:.4f}" for col, val in metrics.iloc[0].items()])
            self.log(f"EVALUATION: {metrics_str}")
            
            # Plotting
            test_start_pos = len(train_val_df)
            test_dates = original_dates[test_start_pos : test_start_pos + len(test_df)]
            
            self.root.after(0, lambda: self.update_chart(test_dates, y_true, y_pred))
            self.root.after(0, lambda: self.btn_predict.config(state="normal"))
            
        except Exception as e:
            self.log(f"PREDICTION ERROR: {str(e)}")
            import traceback
            traceback.print_exc()
            self.root.after(0, lambda: self.btn_predict.config(state="normal"))

    def update_chart(self, dates, y_true, y_pred):
        self.ax.clear()
        
        self.ax.plot(dates, y_true, label="Actual Price", color='blue', alpha=0.6)
        self.ax.plot(dates, y_pred, label="Predicted (Stacking)", color='red', alpha=0.8)
        
        self.ax.set_title("Stock Price Prediction Results")
        self.ax.set_xlabel("Date")
        self.ax.set_ylabel("Price (USD)")
        self.ax.legend()
        self.ax.grid(True)
        
        # Rotate date labels
        plt.setp(self.ax.get_xticklabels(), rotation=45, ha="right")
        self.fig.tight_layout()
        
        self.canvas.draw()
        self.log("Chart updated successfully.")

if __name__ == "__main__":
    root = tk.Tk()
    app = StockPredictionApp(root)
    root.mainloop()
