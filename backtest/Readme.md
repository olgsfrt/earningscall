# Install Zipline
The following steps are required to setup a new environment for the zipline backtester (https://github.com/quantopian/zipline)
zipline is quite picky with regard to version numbers of the packages it depends on - so we better set up a dedicated environment

	conda create -n backtester python=3.5
	conda activate backtester
	conda install -c Quantopian zipline

# Create bundle from Eikon data
To create a zipline bundle from Eikon data, use the following command:
    zipline -e backtest/zipline_ingest.py ingest -b 'eikon-data-bundle'

# To clean old bundles
    zipline -e backtest/zipline_ingest.py clean -b 'eikon-data-bundle' --keep-last 1