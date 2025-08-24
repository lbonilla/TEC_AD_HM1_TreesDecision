    @staticmethod
    def calculate_gini(data_partition_torch, num_classes=2):
        if data_partition_torch.numel() == 0:
            return 0.0
        labels = data_partition_torch[:, -1].long()
        # Contar cuántos ejemplos hay de cada clase
        counts = torch.bincount(labels, minlength=num_classes)
        probs = counts.float() / counts.sum()
        gini = 1.0 - torch.sum(probs ** 2)
        return gini.item()