###-----------------------------------------------------------------------------
### Unit Tests función: calculate_gini
###-----------------------------------------------------------------------------

# Prueba unitaria 1: Gini para un solo grupo homogéneo (impureza debe ser 0)
def test_gini_homogeneous():
    node = Node_CART(num_classes=2)
    # Todos los elementos son de la clase 0
    data = torch.tensor([[1, 2, 0], [2, 3, 0], [3, 4, 0]], dtype=torch.float32)
    gini = node.calculate_gini(data, num_classes=2)
    print('Test 1 - Gini homogéneo:', gini)
    assert abs(gini - 0.0) < 1e-6, f"Esperado 0.0, obtenido {gini}"

# Prueba unitaria 2: Gini para dos clases balanceadas (impureza máxima)
def test_gini_balanced():
    node = Node_CART(num_classes=2)
    # Mitad clase 0, mitad clase 1
    data = torch.tensor([[1, 2, 0], [2, 3, 1], [3, 4, 0], [4, 5, 1]], dtype=torch.float32)
    gini = node.calculate_gini(data, num_classes=2)
    print('Test 2 - Gini balanceado:', gini)
    assert abs(gini - 0.5) < 1e-6, f"Esperado 0.5, obtenido {gini}"

# Ejecutar pruebas
print(" *** Unit test de calculate_gini ")
test_gini_homogeneous()
test_gini_balanced()

def test_select_best_feature_and_thresh():
    # El feature 0 separa perfectamente las clases con threshold 2
    # [feature0, feature1, clase]
    data = torch.tensor([
        [1.0, 10.0, 0],
        [2.0, 20.0, 0],
        [3.0, 30.0, 1],
        [4.0, 40.0, 1]
    ])

    node = Node_CART(num_classes=2)
    best_feature, best_thresh, best_gini = node.select_best_feature_and_thresh(data, num_classes=2)

    print("Mejor feature:", best_feature)
    print("Mejor threshold:", best_thresh)
    print("Mejor gini:", best_gini)


    # Esperamos que el mejor feature sea 0 y el mejor threshold sea 3.0 (ambos separan perfectamente)
    assert best_feature == 0, "El mejor feature debería ser la columna 0"
    assert best_thresh == 3, "El mejor threshold debería ser  3.0"
    assert best_gini == 0.0, "El gini debería ser 0 para una separación perfecta"

###-----------------------------------------------------------------------------
### Unit Tests función: select_best_feature_and_thresh
###-----------------------------------------------------------------------------

def test_select_best_feature_and_thresh_feature2():
    # Dataset: solo la columna 2 permite separación perfecta con threshold 12.5
    # [feature0, feature1, feature2, clase]
    data = torch.tensor([
        [0.0, 1.0, 10.0, 1],
        [0.0, 1.0, 11.0, 1],
        [0.0, 1.0, 12.0, 1],
        [0.0, 1.0, 13.0, 0],
        [0.0, 1.0, 14.0, 0],
        [0.0, 1.0, 15.0, 0],
        [0.0, 1.0, 16.0, 0],
        [0.0, 1.0, 17.0, 0],
    ])

    node = Node_CART(num_classes=2)
    best_feature, best_thresh, best_gini = node.select_best_feature_and_thresh(data, num_classes=2)

    print("Mejor feature:", best_feature)
    print("Mejor threshold:", best_thresh)
    print("Mejor gini:", best_gini)

    # Esperamos que el mejor feature sea 2 (columna 2) y threshold  13.0
    assert best_feature == 2, "El mejor feature debería ser la columna 2"

    assert best_thresh == 13.0 , "El mejor threshold debería ser entre 13"
    assert best_gini == 0.0, "El gini debería ser 0 para una separación perfecta"

print(" *** Unit test de select_best_feature_and_thresh ")
print("Test #1 de la función select_best_feature_and_thresh. El feature 0 separa perfectamente las clases con threshold 2.5")
test_select_best_feature_and_thresh()

print("_____________________________________________________")

print("Test #2 de la función select_best_feature_and_thresh. El feature 2 separa perfectamente las clases con threshold 12.5")
test_select_best_feature_and_thresh_feature2()