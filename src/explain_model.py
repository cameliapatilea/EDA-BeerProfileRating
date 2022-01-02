import lime
import shap
from lime import lime_tabular


def explain_model(model, X_train, y_train, X_test, y_test, explainability_method="shap", task_type: str = "classification"):
    idx = 13
    # https://coderzcolumn.com/tutorials/machine-learning/how-to-use-lime-to-understand-sklearn-models-predictions
    # https://towardsdatascience.com/explainable-ai-xai-with-shap-regression-problem-b2d63fdca670
    if explainability_method == "shap":
        if task_type == "regression":
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            shap.summary_plot(shap_values, X_train, feature_names=features, plot_type="bar")
            shap.summary_plot(shap_values, X_train, feature_names=features)
            shap.dependence_plot(5, shap_values, X_train, feature_names=features)
            shap.force_plot(explainer.expected_value, shap_values[idx], X_test[idx], feature_names=features)
            shap.force_plot(explainer.expected_value, shap_values[0:5, :], X_test[0:5, :], plot_cmap="DrDb", feature_names=features)
            shap.decision_plot(explainer.expected_value[0], shap_values[0], feature_names=list(features))
        elif task_type == "classification":
            pass
        else:
            raise Exception("Wrong task_type given!")
    elif explainability_method == "lime":
        if task_type == "classification":
            explainer = lime_tabular.LimeTabularExplainer(X_train, mode="regression", feature_names=boston.feature_names)
            print(explainer)
            print("Prediction : ", model.predict(X_test[idx].reshape(1, -1)))
            print("Actual :     ", Y_test[idx])

            explanation = explainer.explain_instance(X_test[idx], model.predict, num_features=len(boston.feature_names))
            print(explanation)

            explanation.show_in_notebook()

            with plt.style.context("ggplot"):
                explanation.as_pyplot_figure()

            with plt.style.context("ggplot"):
                fig = plt.figure(figsize=(8, 5))
                plt.barh(range(len(model.coef_)), model.coef_, color=["red" if coef < 0 else "green" for coef in model.coef_])
                plt.yticks(range(len(model.coef_)), boston.feature_names)
                plt.title("Weights")

            print(explanation.as_list())

            print(explanation.as_map())
        elif task_type == "classification":
            explainer = lime_tabular.LimeTabularExplainer(X_train, mode="classification",
                                                          class_names=wine.target_names,
                                                          feature_names=wine.feature_names,
                                                          )

            print(explainer)

            print("Prediction : ", wine.target_names[gb.predict(X_test[idx].reshape(1, -1))[0]])
            print("Actual :     ", wine.target_names[Y_test[idx]])

            explanation = explainer.explain_instance(X_test[idx], gb.predict_proba, top_labels=3)

            explanation.show_in_notebook()

            print("Prediction : ", wine.target_names[model.predict(X_test[idx].reshape(1, -1))[0]])
            print("Actual :     ", wine.target_names[Y_test[idx]])

            explanation = explainer.explain_instance(X_test[idx], gb.predict_proba, top_labels=3)

            explanation.show_in_notebook()

            print("Explanation Local Prediction              : ", explanation.local_pred)
            print("Explanation Global Prediction Probability : ", explanation.predict_proba)
            print("Explanation Global Prediction             : ", wine.target_names[np.argmax(explanation.predict_proba)])
        else:
            raise Exception("Wrong task_type given!")
    else:
        raise Exception("Wrong explainability_method given!")
