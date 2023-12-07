
1.目的：實作IR系統以實現不同Ranking方法，檢索前10個結果和分數。
(1)TF-IDF Weighting + Cosine Similarity
(2)TF-IDF Weighting + Euclidean Distance
(3)Relevance Feedback = [1 * original query + 0.5 * feedback query]

2.方法：使用reuters.com的7,875條英語新聞，每個文件均以其新聞 ID 命名，並包含相應的新聞標題和內容。

3.指標評估實作IR系統。
(1)Recall@10
(2)MAP@10
(3)MRR@10
