import yfinance as yf
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pandas as pd
import tkinter as tk
from tkinter import messagebox, Listbox, Checkbutton, IntVar
from datetime import datetime, timedelta

# Global variables for portfolio and zooming
options = None
r = None
sigma = None
time_today = None
time_halfway = None
time_expiry = None
real_time_stock_price = None
underlying_symbol = None
current_x_range = None
fig, ax = None, None

# Black-Scholes formula for calculating the option price
def black_scholes(S, K, T, r, sigma, option_type='call'):
    if T <= 0:
        if option_type == 'call':
            return max(S - K, 0)  # Intrinsic value for call option
        elif option_type == 'put':
            return max(K - S, 0)  # Intrinsic value for put option
    
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    
    if option_type == 'call':
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    elif option_type == 'put':
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    else:
        raise ValueError("Invalid option type. Choose 'call' or 'put'.")
    
    return price

# Function to fetch real-time stock prices
def get_real_time_stock_price(ticker):
    stock = yf.Ticker(ticker)
    stock_info = stock.history(period="1d")
    if stock_info.empty:
        return None
    return stock_info['Close'].iloc[-1]  # Latest close price

# Function to automatically load the latest CSV file
def load_latest_file():
    downloads_folder = os.path.join(os.path.expanduser('~'), 'Downloads')
    files = [f for f in os.listdir(downloads_folder) if f.startswith('tastytrade_positions') and f.endswith('.csv')]

    if not files:
        raise FileNotFoundError("No files starting with 'tastytrade_positions' found in the Downloads folder.")
    
    latest_file = max(files, key=lambda f: os.path.getmtime(os.path.join(downloads_folder, f)))
    latest_file_path = os.path.join(downloads_folder, latest_file)

    try:
        df = pd.read_csv(latest_file_path)
        return df
    except Exception as e:
        raise IOError(f"Failed to load the CSV file: {e}")

# Function to display trades when a symbol is selected
def display_trades_for_symbol(event):
    global symbol_options_data, checkbox_vars, checkboxes_frame

    selected_symbol_idx = listbox_symbols.curselection()
    if not selected_symbol_idx:
        messagebox.showwarning("No Selection", "Please select a symbol to view trades.")
        return

    selected_symbol = listbox_symbols.get(selected_symbol_idx)

    # Filter the data to get all positions (options and stock) for the selected symbol
    symbol_options_data = pd.concat([
        options_data[options_data['Symbol'].str.startswith(selected_symbol)],
        stock_data[stock_data['Symbol'] == selected_symbol]
    ])

    # Clear the checkbox frame and add checkboxes for the selected trades
    for widget in checkboxes_frame.winfo_children():
        widget.destroy()

    checkbox_vars = []
    for index, row in symbol_options_data.iterrows():
        var = IntVar(value=1)
        checkbox_vars.append(var)

        if row['Type'] == 'STOCK':
            check_text = f"Stock: {row['Symbol']} Position: {row['Quantity']}"
        elif row['Type'] == 'OPTION':
            call_put = row['Call/Put'] if pd.notna(row['Call/Put']) else 'Unknown'
            strike = row['Strike Price'] if pd.notna(row['Strike Price']) else 'Unknown'
            exp_date = row['Exp Date'] if pd.notna(row['Exp Date']) else 'Unknown'
            check_text = f"Option: {call_put} {strike} {exp_date} Position: {row['Quantity']}"

        check = Checkbutton(checkboxes_frame, text=check_text, variable=var)
        check.pack(anchor="w")

# Load the latest CSV and calculate the portfolio
def load_and_display_symbols():
    global options_data, stock_data, symbol_options_data
    trades_df = load_latest_file()

    if trades_df is None:
        return

    options_data = trades_df[trades_df['Type'] == 'OPTION']
    stock_data = trades_df[trades_df['Type'] == 'STOCK']

    options_underlyings = options_data['Symbol'].str.split(' ').str[0]
    stock_underlyings = stock_data['Symbol']

    all_underlyings = pd.concat([options_underlyings, stock_underlyings]).unique()

    listbox_symbols.delete(0, tk.END)
    for underlying in all_underlyings:
        listbox_symbols.insert(tk.END, underlying)

# Function to calculate the portfolio value and P&L over time
def portfolio_value_over_time(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry):
    values_today = np.zeros(len(stock_price_range))
    values_halfway = np.zeros(len(stock_price_range))
    values_expiry = np.zeros(len(stock_price_range))

    for i, stock_price in enumerate(stock_price_range):
        for option in options:
            trade_price = abs(option['trade_price'])  # Use absolute value to handle debits/credits
            position = option['position']
            option_type = option['type']
            strike_price = option['strike']

            if option_type == 'stock':
                if option['trade_price'] < 0:
                    stock_pl = (stock_price - trade_price) * position
                else:
                    stock_pl = (trade_price - stock_price) * position

                values_today[i] += stock_pl
                values_halfway[i] += stock_pl
                values_expiry[i] += stock_pl
            else:
                # Calculate P&L for today using Black-Scholes
                option_value_today = black_scholes(stock_price, strike_price, time_today, r, sigma, option_type)
                option_pl_today = (option_value_today * 100 - trade_price * 100) * position
                values_today[i] += option_pl_today

                # Calculate P&L for halfway using Black-Scholes
                option_value_halfway = black_scholes(stock_price, strike_price, time_halfway, r, sigma, option_type)
                option_pl_halfway = (option_value_halfway * 100 - trade_price * 100) * position
                values_halfway[i] += option_pl_halfway

                # Calculate P&L at expiry using intrinsic value
                if option_type == 'call':
                    option_value_expiry = max(stock_price - strike_price, 0)  # Intrinsic value for call
                elif option_type == 'put':
                    option_value_expiry = max(strike_price - stock_price, 0)  # Intrinsic value for put

                option_pl_expiry = (option_value_expiry * 100 - trade_price * 100) * position
                values_expiry[i] += option_pl_expiry

    return values_today, values_halfway, values_expiry

# Function to plot the portfolio value over time
def plot_portfolio(selected_trades):
    global options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol

    try:
        r = 0.0525  # Risk-free rate is set to 5.25%
        sigma = 0.2  # Assume 20% volatility

        first_selected_symbol = symbol_options_data.iloc[selected_trades[0]]['Symbol']
        selected_underlying = first_selected_symbol.split(' ')[0]  # Get the stock symbol, e.g., 'AAPL'

        # Fetch real-time stock price
        real_time_stock_price = get_real_time_stock_price(selected_underlying)
        if real_time_stock_price is None:
            messagebox.showerror("Error", f"Could not retrieve real-time stock price for {selected_underlying}.")
            return

        # Generate stock price range from -20% to +20% of the real-time stock price
        stock_price_min = 0.8 * real_time_stock_price
        stock_price_max = 1.2 * real_time_stock_price
        stock_price_range = np.linspace(stock_price_min, stock_price_max, 100)

        # Initialize the options variable with selected trades data
        options = []
        for trade in selected_trades:
            trade_info = symbol_options_data.iloc[trade]
            call_put_value = trade_info.get('Call/Put')

            if pd.isna(call_put_value):  # It's a stock
                option_type = 'stock'
                strike = 0  # No strike price for stock
                time_to_expiry = 0  # No expiration for stocks
            else:  # It's an option
                option_type = str(call_put_value).lower()
                strike = float(trade_info['Strike Price'])
                days_to_expiration = trade_info['Days To Expiration']
                time_to_expiry = int(days_to_expiration.rstrip('d'))  # Convert to integer

            position = int(trade_info['Quantity'])
            trade_price = float(trade_info['Trade Price'])

            options.append({
                'type': option_type,
                'strike': strike,
                'position': position,
                'time_to_expiry': time_to_expiry,
                'trade_price': trade_price
            })

        # Define time values for today, halfway, and expiry
        time_expiry = max([option['time_to_expiry'] for option in options]) / 365.0  # Time to expiry in years
        time_today = 1 / 365.0  # A very small value representing today's value
        time_halfway = time_expiry / 2  # Halfway to expiry

        # Call the initial plot setup with zoom buttons
        initial_plot(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, selected_underlying)

    except Exception as e:
        messagebox.showerror("Error", f"An error occurred: {str(e)}")

# Function to collect selected trades and plot
def on_plot_button_click():
    try:
        selected_trades = [i for i, var in enumerate(checkbox_vars) if var.get() == 1]
        if not selected_trades:
            messagebox.showwarning("No Trades Selected", "Please select at least one trade to plot.")
            return

        # Plot the selected portfolio
        plot_portfolio(selected_trades)
    except Exception as e:
        messagebox.showerror("Error", f"An error occurred in plotting: {str(e)}")

# Zoom in functionality
def zoom_in():
    global current_x_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol
    x_min, x_max = current_x_range
    new_x_min = x_min * 1.05
    new_x_max = x_max * 0.95
    current_x_range = (new_x_min, new_x_max)
    stock_price_range = np.linspace(new_x_min, new_x_max, 100)
    plot_portfolio_with_time(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol)

# Zoom out functionality
def zoom_out():
    global current_x_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol
    x_min, x_max = current_x_range
    new_x_min = x_min * 0.95
    new_x_max = x_max * 1.05
    current_x_range = (new_x_min, new_x_max)
    stock_price_range = np.linspace(new_x_min, new_x_max, 100)
    plot_portfolio_with_time(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol)

# Initial setup and plotting

def initial_plot(stock_price_range, plot_options, plot_r, plot_sigma, plot_time_today, plot_time_halfway, plot_time_expiry, plot_real_time_stock_price, plot_underlying_symbol):
    global current_x_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol, fig, ax
    current_x_range = (stock_price_range[0], stock_price_range[-1])  # Store the initial x-axis range
    
    # Assign the global variables for future use
    options = plot_options
    r = plot_r
    sigma = plot_sigma
    time_today = plot_time_today
    time_halfway = plot_time_halfway
    time_expiry = plot_time_expiry
    real_time_stock_price = plot_real_time_stock_price
    underlying_symbol = plot_underlying_symbol
    
    # Initialize the figure and axis for plotting (with reduced size)
    fig, ax = plt.subplots(figsize=(7, 5))  # Adjusted to half the original size

    # Plot the portfolio initially
    plot_portfolio_with_time(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol)

    # Add buttons for zoom in and zoom out below the graph
    zoom_in_button = plt.axes([0.35, 0.01, 0.1, 0.05])  # Position for the "Zoom In" button
    zoom_out_button = plt.axes([0.55, 0.01, 0.1, 0.05])  # Position for the "Zoom Out" button

    zoom_in_button_obj = plt.Button(zoom_in_button, 'Zoom In')
    zoom_out_button_obj = plt.Button(zoom_out_button, 'Zoom Out')

    # Adjust the font size to 75% of the default
    zoom_in_button_obj.label.set_fontsize(8)
    zoom_out_button_obj.label.set_fontsize(8)

    zoom_in_button_obj.on_clicked(lambda event: zoom_in())
    zoom_out_button_obj.on_clicked(lambda event: zoom_out())

    plt.show()






# Function to plot the P&L over three different times with larger graph size and a vertical line for the current stock price
def plot_portfolio_with_time(stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry, real_time_stock_price, underlying_symbol):
    global fig, ax, current_x_range  # Make fig and ax global

    # Initialize the figure and axis if they don't already exist
    if 'fig' not in globals() or 'ax' not in globals() or fig is None or ax is None:
        fig, ax = plt.subplots(figsize=(7, 5))  # Adjusted to half the original size

    else:
        ax.clear()  # Clear the previous graph to redraw with new zoom levels

    # Calculate portfolio values for today, halfway, and expiry
    values_today, values_halfway, values_expiry = portfolio_value_over_time(
        stock_price_range, options, r, sigma, time_today, time_halfway, time_expiry
    )

    # Plot P&L for today (Black)
    ax.plot(stock_price_range, values_today, label='Today', color='black')

    # Plot P&L for halfway (Red)
    ax.plot(stock_price_range, values_halfway, label='Halfway to Expiry', color='red')

    # Plot P&L for expiry (Green)
    ax.plot(stock_price_range, values_expiry, label='At Expiry', color='green')

    expiry_date = None
    halfway_date = None
    for option in options:
        if option['time_to_expiry'] > 0:
            expiry_days = option['time_to_expiry']
            expiry_date = datetime.today() + timedelta(days=expiry_days)
            expiry_date_str = expiry_date.strftime("%d-%b-%y")

            halfway_days = expiry_days / 2
            halfway_date = datetime.today() + timedelta(days=halfway_days)
            halfway_date_str = halfway_date.strftime("%d-%b-%y")
            break

    if expiry_date and halfway_date:
        expiry_label = f"At Expiry: {expiry_date_str}"
        halfway_label = f"Halfway to Expiry: {halfway_date_str}"
    else:
        expiry_label = "Expiry: Unknown"
        halfway_label = "Halfway: Unknown"

    ax.axvline(real_time_stock_price, color='orange', linestyle='--', label=f'Current Price: {real_time_stock_price:.2f}')
    ax.axhline(0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Stock Price')
    ax.set_ylabel('Portfolio P&L')
    ax.set_title(f'Option Portfolio P&L for {underlying_symbol}')
    ax.grid(True, which='both', axis='both', linestyle='--', linewidth=0.7)

    ax.legend([
        'Today',
        halfway_label,
        expiry_label,
        f'Current Price: {real_time_stock_price:.2f}'
    ])

    plt.draw()

    # Close the figure on close event
    fig.canvas.mpl_connect('close_event', lambda event: plt.close())

# Remove the toolbar from the plot
plt.rcParams['toolbar'] = 'none'

# GUI setup for selecting symbols and plotting
root = tk.Tk()
root.title("Option Portfolio Payoff Plotter")

# Center the tkinter window on the screen
window_width = 500
window_height = 400
screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()
position_top = int(screen_height/2 - window_height/2)
position_right = int(screen_width/2 - window_width/2)
root.geometry(f"{window_width}x{window_height}+{position_right}+{position_top}")

tk.Label(root, text="Select Symbol").grid(row=1, column=0, columnspan=2)
listbox_symbols = Listbox(root, selectmode=tk.SINGLE, width=50, height=5)
listbox_symbols.grid(row=2, column=0, columnspan=2)

checkboxes_frame = tk.Frame(root)
checkboxes_frame.grid(row=5, column=0, columnspan=2)

load_and_display_symbols()

listbox_symbols.bind('<<ListboxSelect>>', display_trades_for_symbol)
tk.Button(root, text="Plot Payoff", command=on_plot_button_click).grid(row=6, column=0, columnspan=2, pady=10)

root.mainloop()
