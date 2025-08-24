def test_CART(root_node, D):
    """
    Evalúa un árbol CART previamente entrenado (root_node) sobre un conjunto de datos D (tensor).
    Calcula y retorna la tasa de aciertos (accuracy), definida como:
        accuracy = c / n
    donde:
        c = número de estimaciones correctas (predicción == etiqueta real)
        n = número total de muestras

    Parámetros:
        root_node (Node_CART): nodo raíz del árbol entrenado.
        D (torch.Tensor): conjunto de datos, última columna es la etiqueta.

    Retorna:
        float: tasa de aciertos (accuracy).
    """
    # Contador de aciertos
    correct = 0
    n = D.shape[0]

    # Para cada muestra en D
    for i in range(n):
        sample = D[i, :-1]  # Todas las columnas menos la última (atributos)
        true_label = D[i, -1].item()  # Última columna (etiqueta real)
        pred_label = root_node.evaluate_node(sample)  # Predicción del árbol

        if pred_label == true_label:
            correct += 1

    accuracy = correct / n
    return accuracy
