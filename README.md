# audio_processor
To deal cough audi
| 項目名稱 | 中文說明 | 臨床意義與研究發現 |
|---|---|---|
| Length | 訊號從起始到結束的持續時間，以秒為單位報告。自動分段後，估算訊號最大值低於 -30 dB 前一個結束點為結束，最後一個結束點為結束 | 體聲嘆氣/清痰是否有足夠力度，研究發現，有分吸入(P/A)的患者清痰時間顯著較長 |
| Amplitude contour | 表示訊號包絡的相對強度變化經正規化處理，平均峰值於一小段時間內的變動幅度和前期/後期放大變動的影響 | 咳嗽能量的整體變化趨勢 |
| Amplitude contour slope | 振幅輪廓的斜率趨勢。正係數值代表向下-向上（上的）斜率；負係數值代表向上-向下（下）的斜率，透過第二個傾斜校驗轉換(DCT)係數描述 | 能量上升速度，代表病患清痰/清痰無力指標之一，是否有/無平吸咳嗽患者的關鍵特徵之一 |
| Amplitude contour curvature | 振幅輪廓的彎曲方向。正係數值代表向下-向上（上的）曲率；負係數值代表向上-向下（下）的曲率，透過第三個DCT係數描述 | 咳嗽發是否在其中，對稱。與振幅正/仄化大有關，對P/A有顯著差異。研究顯示，咳嗽患者的輪廓式分佈性咳嗽，輪廓曲率值越低，可能表示情結束時的振幅減弱或缺失 |
| Sample entropy contour | 反映訊號的亂流流率與強度特徵而產生的複雜程度，報告訊號的不均勻規則或混亂。它能區分無流量和有氣流的聲掛指標 | 變音中的複雜程度，反映氣流不穩或聲門漏氣 |
| Kurtosis contour | 報告訊號本體的局部直方圖之尖銳程度，報告信號結束時振幅分布。明顯的能量特徵有助於區分音或氣流變動的尖銳度 | 量變突變變動的集中度（是否尖銳）。研究發現，在有顯著差異的患者中，邊緣式反射咳嗽與自然反射呼吸動作Kurtosis contour平均值存在差異 |
| Crest factor | 表示訊號中最大樣本值相對於根均方根值（RMS）較大時的特徵，反映訊號的能量強度與信號性質 | 最大聲量是否明顯較大。研究顯示在P/A病患與非P/A患者中，滿意反射咳嗽與吸氣咳嗽有差異 |
| Relative position of crest factor | 報告波峰位於子區間點中的相對位置（起始/結束/中間） | 反映最大能量出現的時間點，與咳嗽結束時的能量分布有關 |