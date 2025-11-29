# This project calculates how much you can expect to loose on your portfolio in the next day undermarket conditions

# Libraries :
import streamlit
import pandas
import numpy
import yfinance
import matplotlib.pyplot

DEFAULT_ASSETS = 'SPY, TLT, AAPL'
# SPY is the exchange traded fund, TLT is the long term treasury bond EFT, AAPL stands for Apple incorporated

def calculate_historical_var(assets, weights, confidence_level, time_horizon_days):
   
    if not assets:
        streamlit.error("Please enter at least one asset ticker.")
        return None, None

    try:
        assets_string = " ".join(assets)
        data = yfinance.download(assets_string, period="5y", interval="1d", progress=False)

        try:
            if isinstance(data.columns, pandas.MultiIndex):
                data = data['Adj Close']
            else:
                data = data[['Adj Close']]
        except KeyError:
            if isinstance(data.columns, pandas.MultiIndex):
                data = data['Close']
            else:
                data = data[['Close']]

        if isinstance(data.columns, pandas.MultiIndex):
            data.columns = data.columns.droplevel(0)

        data.columns = [col.upper() for col in data.columns]

        if data.empty:
            streamlit.error("Could not download data for the specified assets. Check ticker spelling.")
            return None, None

        log_returns = numpy.log(data / data.shift(1)).dropna()
        weights = numpy.array(weights) / numpy.sum(weights)
        portfolio_returns = log_returns @ weights
        alpha = 1 - confidence_level / 100
        var_daily_return = -portfolio_returns.quantile(alpha)
        var_final = var_daily_return * numpy.sqrt(time_horizon_days)

        return var_final, portfolio_returns

    except Exception as e:
        streamlit.error(f"A final critical error occurred: {e}")
        return None, None

streamlit.set_page_config(layout="wide")
streamlit.title("🛡️ Algorithmic Risk Modeling Tool (Historical VaR)")
streamlit.subheader("Financial Simulation Project using Python")
streamlit.markdown("---")

streamlit.sidebar.header("Inputs & Parameters")

assets_input = streamlit.sidebar.text_input(
    "Asset Tickers (e.g., SPY, TLT, AAPL)",
    DEFAULT_ASSETS
).split(',')

assets = [t.strip().upper() for t in assets_input if t.strip()]

weights_input = []
if assets:
    streamlit.sidebar.markdown(f"**Define Weights (Total should be 100%):**")
    total_weight = 0
    
    cols = streamlit.sidebar.columns(len(assets))
    for i, asset in enumerate(assets):
        try:
            default_weight = 100 / len(assets) if len(assets) <= 4 else 0
            
            weight = cols[i].number_input(
                f"{asset}",
                min_value=0,
                max_value=100,
                value=int(default_weight), 
                step=1,
                key=f"weight_{asset}"
            )
            weights_input.append(weight / 100)
            total_weight += weight

        except Exception:
            weights_input.append(0)
            
    streamlit.sidebar.info(f"Current Total Weight: {total_weight}%")

else:
    streamlit.sidebar.warning("No valid tickers entered.")

streamlit.sidebar.markdown("---")
confidence_level = streamlit.sidebar.slider(
    "Confidence Level (%)",
    min_value=90,
    max_value=99,
    value=95,
    step=1
)

time_horizon_days = streamlit.sidebar.number_input(
    "Time Horizon (Days)",
    min_value=1,
    max_value=252, 
    value=1,
    step=1
)

if streamlit.button("Calculate VaR"):
    if assets and numpy.sum(weights_input) > 0:
        
        with streamlit.spinner('Downloading data and calculating VaR...'):
            var_result, returns_data = calculate_historical_var(
                assets, 
                weights_input, 
                confidence_level, 
                time_horizon_days
            )

        if var_result is not None:
            col1, col2 = streamlit.columns(2)
            
            with col1:
                streamlit.metric(
                    label=f"💰 Value-at-Risk ({confidence_level}%)",
                    value=f"{var_result * 100:.2f}%",
                    help=f"Max expected loss over {time_horizon_days} days with {confidence_level}% confidence.")
            
            with col2:
                 streamlit.metric(
                    label="Time Horizon",
                    value=f"{time_horizon_days} Days")

            streamlit.markdown("---")
        
            if returns_data is not None:
                streamlit.header("Portfolio Return Distribution")
                
                fig, ax = matplotlib.pyplot.subplots(figsize=(10, 6))
                
                returns_data_percent = returns_data * 100
                returns_data_percent.hist(ax=ax, bins=50, density=True, alpha=0.6, color='skyblue')
                
                var_line_return = -var_result / numpy.sqrt(time_horizon_days) * 100
                ax.axvline(var_line_return, color='red', linestyle='dashed', linewidth=2, 
                           label=f'VaR ({confidence_level}%)')
                
                ax.set_title("Historical Portfolio Daily Return Distribution")
                ax.set_xlabel("Daily Log Return (%)")
                ax.set_ylabel("Frequency (Density)")
                ax.legend()
                streamlit.pyplot(fig)
                
                streamlit.caption(f"The red dashed line represents the return at which the expected loss is exceeded only {100 - confidence_level}% of the time.")
