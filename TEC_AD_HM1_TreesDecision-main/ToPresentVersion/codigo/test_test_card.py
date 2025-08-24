###-----------------------------------------------------------------------------
### Unit Tests función: test_card
###-----------------------------------------------------------------------------


def test_CART_simple_CART():
    # Dataset perfectamente separable por feature 0 con threshold 2.0
    data = torch.tensor([
        [1.0, 10.0, 0],
        [2.0, 20.0, 0],
        [3.0, 30.0, 1],
        [4.0, 40.0, 1]
    ])

    node = Node_CART(num_classes=2)
    node.create_with_children(data, current_depth=0, min_gini=0.0)
    acc = test_CART(node, data)  # accuracy
    print("Accuracy árbol perfectamente valanceado:", acc)
    assert acc == 1.0, "El accuracy debería ser 1"

def test_CART_DEPTH_2():
  data = torch.tensor([
      [1.0, 0.0, 1.0, 0],
      [1.0, 0.0, 2.0, 0],
      [1.0, 0.0, 3.0, 1],
      [1.0, 0.0, 4.0, 0],
      [1.0, 0.0, 5.0, 0],
      [1.0, 0.0, 6.0, 0],
  ])

  node = Node_CART(num_classes=2)
  node.create_with_children(data, current_depth=0)
  acc = test_CART(node, data)  # accuracy
  print("Accuracy árbol usando un min gini mayor a 0", acc)
  assert acc < 1.0 and acc >0, "El accuracy debería ser menor que uno y mayor a 0, usando un min gini mayor a 0"

def test_CART_DEPTH_2_Gin0():
  data = torch.tensor([
      [1.0, 0.0, 1.0, 0],
      [1.0, 0.0, 2.0, 0],
      [1.0, 0.0, 3.0, 1],
      [1.0, 0.0, 4.0, 0],
      [1.0, 0.0, 5.0, 0],
      [1.0, 0.0, 6.0, 0],
  ])

  node = Node_CART(num_classes=2)
  node.create_with_children(data, current_depth=0, min_gini=0)
  acc = test_CART(node, data)  # accuracy
  print("Accuracy árbol usando un min gini 0", acc)
  assert acc < 1 , "El accuracy debería ser igual a 1"

print(" *** Unit test de test_card ")
test_CART_simple_CART()
test_CART_DEPTH_2()
test_CART_DEPTH_2_Gin0()