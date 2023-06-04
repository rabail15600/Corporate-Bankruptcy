# Corporate Bankruptcy
 
# Introduction to problem

We have taken a quest of identifying financial distressed listed companies
that can go bankrupt to help financial institutions make appropriate lending
decisions. The loans extended to corporations represent a significant amount
of assets for banks. These loans are financed from deposits which are the 
liabilities for banks. An event of higher number of defaults, when  banks lose 
the ability to settle their obligations, can lead to a run on the  bank where 
many clients withdraw their money from the bank because they think it may cease 
its operations. Therefore it is essential to accurately assess risk before 
extending credit to avoid loss. Additionally, the early prediction would allow 
financial institutions sufficient time to take credit management measures to
limit their economic losses.

# Literature review

The insolvency of a company can have a negative impact on the overall economy. 
A great deal of studies have been conducted to predict bankruptcy to help reduce 
economic loss (Balleisen, 2001; Zywicki, 2008). Prior studies have identified 
that financial ratios are considered as one of the most significant predictors 
in predicting bankruptcy (Altman, 1968; Beaver, 1966; Ohlson, 1980). However, 
several studies have also been conducted identifying the importance of utilizing 
corporate governance indicators to predict bankruptcy (Bredart, 2014; Chen, 
2014; Lee & Yeh, 2004; Lin, Liang, & Chu, 2010; Wu, 2007 ). The paper related to 
this dataset has focused on using financial ratios in association with corporate
governance indicators to produce the best model that identifies financial 
distress. Using the concepts and theories learned in the classroom of Applied 
Statistics for Data Science. I would filter out unrepresentative features from 
the dataset to find the most significant financial features.

# Dataset description:

The dataset was collected from the Taiwan Economic Journal for the years 
1999-2009. Company bankruptcy was defined based on the business regulations 
of the Taiwan Stock Exchange. There were two criteria used to select companies. 
Firstly, the selected companies should have at least three years of public 
financial disclosure before the financial crisis. Secondly, there should be a 
sufficient number of comparable companies at a similar scale for comparison of 
the bankrupt and non-bankrupt cases. The final raw dataset has 6,819 rows with 
95 predictors and a binary classified target indicating if the company can go 
bankrupt or not. Of the 95 predictors, two of the variables are categorical 
while 93 attributes are financial ratios of six different types including 
solvency, profitability, cash flow, capital structure, turnover, and others. 
The final sample of companies that we will be working on includes companies from 
the manufacturing industry comprising industrial and electronics companies 
(346), the service industry composed of shipping, tourism, and retail companies 
(39), and others (93), but not financial companies.

# Target variable:

The target variable is the binary classification of bankruptcy based on the 95 
predictor variables.

# Variable information

•	Y - Bankrupt?: Class label (Categorical)

•	X1 - ROA(C) before interest and depreciation before interest: Return On 
       Total Assets(C) (Numeric)

•	X2 - ROA(A) before interest and % after tax: Return On Total Assets(A) (Numeric)

•	X3 - ROA(B) before interest and depreciation after tax: Return On Total 
       Assets(B) (Numeric)

•	X4 - Operating Gross Margin: Gross Profit/Net Sales (Numeric)

•	X5 - Realized Sales Gross Margin: Realized Gross Profit/Net Sales (Numeric)

•	X6 - Operating Profit Rate: Operating Income/Net Sales (Numeric)

•	X7 - Pre-tax net Interest Rate: Pre-Tax Income/Net Sales (Numeric)

•	X8 - After-tax net Interest Rate: Net Income/Net Sales (Numeric)

•	X9 - Non-industry income and expenditure/revenue: Net Non-operating Income 
       Ratio (Numeric)

•	X10 - Continuous interest rate (after tax): Net Income-Exclude Disposal Gain 
        or Loss/Net Sales (Numeric)

•	X11 - Operating Expense Rate: Operating Expenses/Net Sales (Numeric)

•	X12 - Research and development expense rate: (Research and Development 
        Expenses)/Net Sales (Numeric)

•	X13 - Cash flow rate: Cash Flow from Operating/Current Liabilities (Numeric)

•	X14 - Interest-bearing debt interest rate: Interest-bearing Debt/Equity (Numeric)

•	X15 - Tax rate (A): Effective Tax Rate (Numeric)

•	X16 - Net Value Per Share (B): Book Value Per Share(B) (Numeric)

•	X17 - Net Value Per Share (A): Book Value Per Share(A) (Numeric)

•	X18 - Net Value Per Share (C): Book Value Per Share(C) (Numeric)

•	X19 - Persistent EPS in the Last Four Seasons: EPS-Net Income (Numeric)

•	X20 - Cash Flow Per Share (Numeric)

•	X21 - Revenue Per Share (Yuan ¥): Sales Per Share (Numeric)

•	X22 - Operating Profit Per Share (Yuan ¥): Operating Income Per Share (Numeric)

•	X23 - Per Share Net profit before tax (Yuan ¥): Pretax Income Per Share (Numeric)

•	X24 - Realized Sales Gross Profit Growth Rate (Numeric)

•	X25 - Operating Profit Growth Rate: Operating Income Growth (Numeric)

•	X26 - After-tax Net Profit Growth Rate: Net Income Growth (Numeric)

•	X27 - Regular Net Profit Growth Rate: Continuing Operating Income after Tax 
        Growth (Numeric)

•	X28 - Continuous Net Profit Growth Rate: Net Income-Excluding Disposal Gain or 
        Loss Growth (Numeric)

•	X29 - Total Asset Growth Rate: Total Asset Growth (Numeric)

•	X30 - Net Value Growth Rate: Total Equity Growth (Numeric)

•	X31 - Total Asset Return Growth Rate Ratio: Return on Total Asset Growth (Numeric)

•	X32 - Cash Reinvestment %: Cash Reinvestment Ratio (Numeric)

•	X33 - Current Ratio (Numeric)

•	X34 - Quick Ratio: Acid Test (Numeric)

•	X35 - Interest Expense Ratio: Interest Expenses/Total Revenue (Numeric)

•	X36 - Total debt/Total net worth: Total Liability/Equity Ratio (Numeric)

•	X37 - Debt ratio %: Liability/Total Assets (Numeric)

•	X38 - Net worth/Assets: Equity/Total Assets (Numeric)

•	X39 - Long-term fund suitability ratio (A): (Long-term Liability+Equity)/Fixed 
        Assets (Numeric)

•	X40 - Borrowing dependency: Cost of Interest-bearing Debt (Numeric)

•	X41 - Contingent liabilities/Net worth: Contingent Liability/Equity (Numeric)

•	X42 - Operating profit/Paid-in capital: Operating Income/Capital (Numeric)

•	X43 - Net profit before tax/Paid-in capital: Pretax Income/Capital (Numeric)

•	X44 - Inventory and accounts receivable/Net value: (Inventory+Accounts 
        Receivables)/Equity (Numeric)

•	X45 - Total Asset Turnover (Numeric)

•	X46 - Accounts Receivable Turnover (Numeric)

•	X47 - Average Collection Days: Days Receivable Outstanding (Numeric)

•	X48 - Inventory Turnover Rate (times) (Numeric)

•	X49 - Fixed Assets Turnover Frequency (Numeric)

•	X50 - Net Worth Turnover Rate (times): Equity Turnover (Numeric)

•	X51 - Revenue per person: Sales Per Employee (Numeric)

•	X52 - Operating profit per person: Operation Income Per Employee (Numeric)

•	X53 - Allocation rate per person: Fixed Assets Per Employee (Numeric)

•	X54 - Working Capital to Total Assets (Numeric)

•	X55 - Quick Assets/Total Assets (Numeric)

•	X56 - Current Assets/Total Assets (Numeric)

•	X57 - Cash/Total Assets (Numeric)

•	X58 - Quick Assets/Current Liability (Numeric)

•	X59 - Cash/Current Liability (Numeric)

•	X60 - Current Liability to Assets (Numeric)

•	X61 - Operating Funds to Liability (Numeric)

•	X62 - Inventory/Working Capital (Numeric)

•	X63 - Inventory/Current Liability (Numeric)

•	X64 - Current Liabilities/Liability (Numeric)

•	X65 - Working Capital/Equity (Numeric)

•	X66 - Current Liabilities/Equity (Numeric)

•	X67 - Long-term Liability to Current Assets (Numeric)

•	X68 - Retained Earnings to Total Assets (Numeric)

•	X69 - Total income/Total expense (Numeric)

•	X70 - Total expense/Assets (Numeric)

•	X71 - Current Asset Turnover Rate: Current Assets to Sales (Numeric)

•	X72 - Quick Asset Turnover Rate: Quick Assets to Sales (Numeric)

•	X73 - Working capitcal Turnover Rate: Working Capital to Sales (Numeric)

•	X74 - Cash Turnover Rate: Cash to Sales (Numeric)

•	X75 - Cash Flow to Sales (Numeric)

•	X76 - Fixed Assets to Assets (Numeric)

•	X77 - Current Liability to Liability (Numeric)

•	X78 - Current Liability to Equity (Numeric)

•	X79 - Equity to Long-term Liability (Numeric)

•	X80 - Cash Flow to Total Assets (Numeric)

•	X81 - Cash Flow to Liability (Numeric)

•	X82 - CFO to Assets (Numeric)\

•	X83 - Cash Flow to Equity (Numeric)

•	X84 - Current Liability to Current Assets (Numeric)

•	X85 - Liability-Assets Flag: 1 if Total Liability exceeds Total Assets, 
        0 otherwise (Categorical)

•	X86 - Net Income to Total Assets (Numeric)

•	X87 - Total assets to GNP price (Numeric)

•	X88 - No-credit Interval (Numeric)

•	X89 - Gross Profit to Sales (Numeric)

•	X90 - Net Income to Stockholder's Equity (Numeric)

•	X91 - Liability to Equity (Numeric)

•	X92 - Degree of Financial Leverage (DFL) (Numeric)

•	X93 - Interest Coverage Ratio (Interest expense to EBIT) (Numeric)

•	X94 - Net Income Flag: 1 if Net Income is Negative for the last two years, 0 
        otherwise (Categorical)

•	X95 - Equity to Liability (Numeric)

# Research questions

The questions that we are aiming to answer include:

Q1. Will the company go bankrupt?
Q2. What features or types of ratios are significant in predicting bankruptcy?

With these questions answered, we would be helping all the stakeholders of 
financial institutions (customers, shareholders, management, etc) to minimize
the economic fallout of the event of bankruptcy.

# Business value

The early prediction of bankruptcy not only helps the financial institutions 
make an appropriate lending decision but also forces the company itself to take 
corrective actions. With the identification of financial distress, the credit 
rating agencies take action accordingly and revise the credit risk rating of the 
company, which then aids investors in making better investment decisions and 
minimizing the loss. Moreover, it is also beneficial to identify the industry 
having the most bankrupt cases. This would provide a heads-up to the 
policymakers to address any systematic issues causing bankruptcies.

# Statistical methods

## Partitioning plan:

The classes of "Bankruptcy" are imbalanced thus, the aim is to utilize stratified
sampling to create balanced training and testing set. However, the classifier
model will not distinguish between 0 and 1 well because bankruptcy cases are just
3%. Therefore, we will be under sampling the training data to take the proportion
of bankruptcy cases of 10% versus 3% in the original dataset. Meanwhile, the 
testing set would reflect the proportion of bankruptcy classes similar to the 
actual data to reflect real-word scenario for test model comparison.

## Feature selection:

The aim of the feature selection is to discard the redundant predictor variables
to keep the model as simple as possible. We will be using stepwise logistic 
regression (SLR) to select the most significant variables. Other algorithms
selected also help determine which variables are most important e.g. Gini Index
in Decision tree.

## Modeling work:

There are many models and techniques that can be leveraged to develop predictive
models of bankruptcy. Within the coursework of Applied Statistics for Data
Science, we will be using four classification techniques namely Logistic 
regression, Decision tree, Random forest, and Gradient boosting.

The rationale behind choosing logistic regression is that it is easier to 
implement, interpret, and very efficient to train. It makes no assumptions
about distributions of classes of the features. Additionally, it has good
explainability because it provides coefficient size with its direction (positive
or negative). As for the decision tree, it has built-in variable selection 
mechanism and has no assumptions of linear relationship thus, it is easy to use. 
Importantly, it is a white-box model which means if a given situation is 
observable in a model, the explanation for the condition is easily explained by 
Boolean logic. With regards to random forest and xgboost, they are known to 
perform well in high dimensional settings without being subject to overfitting 
because they are ensembles. Additionally, They are widely used in the banking
industry to predict creditworthiness of a loan applicant.
