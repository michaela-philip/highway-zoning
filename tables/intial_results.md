\begin{center}
\begin{tabular}{lclc}
\toprule
\textbf{Dep. Variable:}                &       hwy        & \textbf{  R-squared:         } &     0.024   \\
\textbf{Model:}                        &       OLS        & \textbf{  Adj. R-squared:    } &     0.021   \\
\textbf{Method:}                       &  Least Squares   & \textbf{  F-statistic:       } &     9.062   \\
\textbf{Date:}                         & Mon, 07 Jul 2025 & \textbf{  Prob (F-statistic):} &  1.52e-08   \\
\textbf{Time:}                         &     14:44:32     & \textbf{  Log-Likelihood:    } &    329.31   \\
\textbf{No. Observations:}             &        2321      & \textbf{  AIC:               } &    -646.6   \\
\textbf{Df Residuals:}                 &        2315      & \textbf{  BIC:               } &    -612.1   \\
\textbf{Df Model:}                     &           5      & \textbf{                     } &             \\
\textbf{Covariance Type:}              &       HC3        & \textbf{                     } &             \\
\bottomrule
\end{tabular}
\begin{tabular}{lcccccc}
                                       & \textbf{coef} & \textbf{std err} & \textbf{z} & \textbf{P$> |$z$|$} & \textbf{[0.025} & \textbf{0.975]}  \\
\midrule
\textbf{Intercept}                     &       0.0621  &        0.022     &     2.791  &         0.005        &        0.018    &        0.106     \\
\textbf{mblack\_mean\_pct}             &      -0.0569  &        0.023     &    -2.430  &         0.015        &       -0.103    &       -0.011     \\
\textbf{Residential}                   &      -0.0434  &        0.021     &    -2.070  &         0.038        &       -0.084    &       -0.002     \\
\textbf{mblack\_mean\_pct:Residential} &       0.0284  &        0.024     &     1.161  &         0.246        &       -0.020    &        0.076     \\
\textbf{rent}                          &   -1.183e-05  &     1.44e-05     &    -0.824  &         0.410        &       -4e-05    &     1.63e-05     \\
\textbf{valueh}                        &     9.04e-06  &     3.31e-06     &     2.735  &         0.006        &     2.56e-06    &     1.55e-05     \\
\bottomrule
\end{tabular}
\begin{tabular}{lclc}
\textbf{Omnibus:}       & 1935.789 & \textbf{  Durbin-Watson:     } &     1.184  \\
\textbf{Prob(Omnibus):} &   0.000  & \textbf{  Jarque-Bera (JB):  } & 30455.889  \\
\textbf{Skew:}          &   4.142  & \textbf{  Prob(JB):          } &      0.00  \\
\textbf{Kurtosis:}      &  18.694  & \textbf{  Cond. No.          } &  3.57e+04  \\
\bottomrule
\end{tabular}
%\caption{OLS Regression Results}
\end{center}

Notes: \newline
 [1] Standard Errors are heteroscedasticity robust (HC3) \newline
 [2] The condition number is large, 3.57e+04. This might indicate that there are \newline
 strong multicollinearity or other numerical problems.