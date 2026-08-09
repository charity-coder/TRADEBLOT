//+------------------------------------------------------------------+
//|                                       CompositeSignalEngine.mq5  |
//|  MT5 port of the "FX Signal Dial" composite score.               |
//|                                                                   |
//|  Combines four sub-signals into one -100..+100 composite score,  |
//|  exactly mirroring the Python/Streamlit version:                 |
//|    1. Trend     : SMA20 vs SMA50 crossover                        |
//|    2. Momentum  : RSI(14) distance from 50                        |
//|    3. MACD      : histogram sign/magnitude relative to price      |
//|    4. Volatility: Bollinger Band position (mean-reversion bias)   |
//|                                                                   |
//|  Install: copy into MQL5/Indicators/, compile in MetaEditor,      |
//|  then drag onto any chart. Works on any symbol/timeframe.         |
//|                                                                   |
//|  Educational tool only — not financial advice.                   |
//+------------------------------------------------------------------+
#property copyright "Composite Signal Engine"
#property version   "1.00"
#property indicator_separate_window
#property indicator_buffers 4
#property indicator_plots   1

#property indicator_type1   DRAW_COLOR_LINE
#property indicator_color1  clrGray, clrTomato, clrSlateGray, clrMediumSeaGreen
#property indicator_width1  2
#property indicator_label1  "Composite Score"

#property indicator_minimum -100
#property indicator_maximum  100
#property indicator_level1   25
#property indicator_level2   -25
#property indicator_level3   0
#property indicator_levelcolor clrDimGray
#property indicator_levelstyle STYLE_DOT

//--- Inputs -----------------------------------------------------------
input int    InpSmaFast      = 20;     // Fast SMA period (trend)
input int    InpSmaSlow      = 50;     // Slow SMA period (trend)
input int    InpRsiPeriod    = 14;     // RSI period (momentum)
input int    InpMacdFast     = 12;     // MACD fast EMA
input int    InpMacdSlow     = 26;     // MACD slow EMA
input int    InpMacdSignal   = 9;      // MACD signal EMA
input int    InpBbPeriod     = 20;     // Bollinger period
input double InpBbDeviation  = 2.0;    // Bollinger deviation
input double InpThreshold    = 25.0;   // BUY/SELL threshold (+/-)
input bool   InpEnableAlerts = true;   // Alert on threshold cross
input bool   InpAlertOncePerBar = true; // Only one alert per closed bar

//--- Buffers ------------------------------------------------------------
double CompositeBuffer[];
double ColorBuffer[];
double ThresholdUpBuffer[];   // helper series (not plotted) for readability
double ThresholdDnBuffer[];

//--- Indicator handles
int hSmaFast, hSmaSlow, hRsi, hMacd, hBbands;

datetime lastAlertBarTime = 0;

//+------------------------------------------------------------------+
int OnInit()
  {
   SetIndexBuffer(0, CompositeBuffer, INDICATOR_DATA);
   SetIndexBuffer(1, ColorBuffer, INDICATOR_COLOR_INDEX);
   SetIndexBuffer(2, ThresholdUpBuffer, INDICATOR_CALCULATIONS);
   SetIndexBuffer(3, ThresholdDnBuffer, INDICATOR_CALCULATIONS);

   ArraySetAsSeries(CompositeBuffer, true);
   ArraySetAsSeries(ColorBuffer, true);

   PlotIndexSetString(0, PLOT_LABEL, "Composite Score");
   IndicatorSetString(INDICATOR_SHORTNAME, "Composite Signal Engine");
   IndicatorSetInteger(INDICATOR_DIGITS, 2);

   hSmaFast = iMA(_Symbol, _Period, InpSmaFast, 0, MODE_SMA, PRICE_CLOSE);
   hSmaSlow = iMA(_Symbol, _Period, InpSmaSlow, 0, MODE_SMA, PRICE_CLOSE);
   hRsi     = iRSI(_Symbol, _Period, InpRsiPeriod, PRICE_CLOSE);
   hMacd    = iMACD(_Symbol, _Period, InpMacdFast, InpMacdSlow, InpMacdSignal, PRICE_CLOSE);
   hBbands  = iBands(_Symbol, _Period, InpBbPeriod, 0, InpBbDeviation, PRICE_CLOSE);

   if(hSmaFast == INVALID_HANDLE || hSmaSlow == INVALID_HANDLE || hRsi == INVALID_HANDLE ||
      hMacd == INVALID_HANDLE || hBbands == INVALID_HANDLE)
     {
      Print("CompositeSignalEngine: failed to create one or more indicator handles");
      return(INIT_FAILED);
     }

   return(INIT_SUCCEEDED);
  }

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
  {
   IndicatorRelease(hSmaFast);
   IndicatorRelease(hSmaSlow);
   IndicatorRelease(hRsi);
   IndicatorRelease(hMacd);
   IndicatorRelease(hBbands);
  }

//+------------------------------------------------------------------+
double Clip(double v, double lo, double hi)
  {
   if(v < lo) return lo;
   if(v > hi) return hi;
   return v;
  }

//+------------------------------------------------------------------+
int OnCalculate(const int rates_total,
                const int prev_calculated,
                const datetime &time[],
                const double &open[],
                const double &high[],
                const double &low[],
                const double &close[],
                const long &tick_volume[],
                const long &volume[],
                const int &spread[])
  {
   int minBars = MathMax(InpSmaSlow, MathMax(InpMacdSlow + InpMacdSignal, InpBbPeriod)) + 5;
   if(rates_total < minBars)
      return(0);

   // Copy indicator values for the whole visible range each call.
   // (Simple & robust; MT5 caches handle results so repeated CopyBuffer
   // calls across ticks are cheap.)
   int need = rates_total;

   double smaFast[], smaSlow[], rsiVal[], macdHist[], bbMid[], bbUpper[], bbLower[];
   if(CopyBuffer(hSmaFast, 0, 0, need, smaFast) <= 0) return(0);
   if(CopyBuffer(hSmaSlow, 0, 0, need, smaSlow) <= 0) return(0);
   if(CopyBuffer(hRsi, 0, 0, need, rsiVal) <= 0) return(0);
   if(CopyBuffer(hMacd, 1, 0, need, macdHist) <= 0) return(0); // MACD histogram buffer
   if(CopyBuffer(hBbands, 0, 0, need, bbMid) <= 0) return(0);
   if(CopyBuffer(hBbands, 1, 0, need, bbUpper) <= 0) return(0);
   if(CopyBuffer(hBbands, 2, 0, need, bbLower) <= 0) return(0);

   ArraySetAsSeries(smaFast, true);
   ArraySetAsSeries(smaSlow, true);
   ArraySetAsSeries(rsiVal, true);
   ArraySetAsSeries(macdHist, true);
   ArraySetAsSeries(bbMid, true);
   ArraySetAsSeries(bbUpper, true);
   ArraySetAsSeries(bbLower, true);

   int start = (prev_calculated > 1) ? rates_total - prev_calculated + 1 : 0;
   // Recompute a small trailing window to stay safe on history edits.
   start = MathMin(start, rates_total - 1);

   for(int shift = start; shift >= 0; shift--)
     {
      int i = rates_total - 1 - shift; // forward index into time-series arrays (0=oldest)
      // Everything below is indexed as "series" (0 = current/most recent bar),
      // so use 'shift' directly against the as-series buffers.
      int s = shift;

      double score = 0.0;
      double votes = 0.0;

      // 1) Trend: SMA fast vs SMA slow
      if(smaSlow[s] != 0.0 && smaFast[s] != EMPTY_VALUE && smaSlow[s] != EMPTY_VALUE)
        {
         double diff = (smaFast[s] - smaSlow[s]) / smaSlow[s];
         score += Clip(diff * 200.0, -1.0, 1.0);
         votes += 1.0;
        }

      // 2) Momentum: RSI distance from 50
      if(rsiVal[s] != EMPTY_VALUE)
        {
         score += Clip((rsiVal[s] - 50.0) / 25.0, -1.0, 1.0);
         votes += 1.0;
        }

      // 3) MACD histogram, normalized by price
      if(macdHist[s] != EMPTY_VALUE && close[rates_total - 1 - s] != 0.0)
        {
         double px = close[rates_total - 1 - s];
         double norm = macdHist[s] / (MathAbs(px) * 0.002 + 0.000000001);
         score += Clip(norm, -1.0, 1.0);
         votes += 1.0;
        }

      // 4) Bollinger position (mild mean-reversion bias)
      if(bbUpper[s] != EMPTY_VALUE && bbLower[s] != EMPTY_VALUE)
        {
         double width = bbUpper[s] - bbLower[s];
         double px = close[rates_total - 1 - s];
         double pos = (width > 0.0) ? (px - bbMid[s]) / (width / 2.0) : 0.0;
         score += Clip(-pos * 0.6, -1.0, 1.0);
         votes += 1.0;
        }

      double composite = (votes > 0.0) ? (score / votes) * 100.0 : 0.0;

      CompositeBuffer[s] = composite;

      if(composite > InpThreshold)
         ColorBuffer[s] = 3; // bull color slot
      else if(composite < -InpThreshold)
         ColorBuffer[s] = 1; // bear color slot
      else
         ColorBuffer[s] = 2; // neutral color slot
     }

   // --- Alerts on the most recently closed bar -------------------------
   if(InpEnableAlerts && rates_total > 1)
     {
      int closedShift = 1; // 0 = forming bar, 1 = last fully closed bar
      double curScore  = CompositeBuffer[closedShift];
      double prevScore = CompositeBuffer[closedShift + 1];
      datetime barTime = time[rates_total - 1 - closedShift];

      bool alreadyAlerted = (InpAlertOncePerBar && barTime == lastAlertBarTime);

      if(!alreadyAlerted)
        {
         if(prevScore <= InpThreshold && curScore > InpThreshold)
           {
            Alert(_Symbol, " ", EnumToString(_Period), ": Composite Signal -> BUY (score ",
                  DoubleToString(curScore, 1), ")");
            lastAlertBarTime = barTime;
           }
         else if(prevScore >= -InpThreshold && curScore < -InpThreshold)
           {
            Alert(_Symbol, " ", EnumToString(_Period), ": Composite Signal -> SELL (score ",
                  DoubleToString(curScore, 1), ")");
            lastAlertBarTime = barTime;
           }
        }
     }

   return(rates_total);
  }
//+------------------------------------------------------------------+
