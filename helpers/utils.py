import random

import torch


def generate_positive_slices(start_idx: int, end_idx: int, n: int, max_length: int) -> list[tuple[int, int]]:
    """
    Возвращает n срезов включающих в себя отрезок [start_idx, end_idx] с некоторой случайностью.
    Длина каждого среза находится между [end_idx - start_idx + 1, max_length].
    Если start_idx и end_idx равны -1, то функция возвращаеь пустой список.

    Аргументы:
        start_idx (int): начальный индекс отрезка
        end_idx (int): конечный индекс отрезка
        n (int): количество срезов, которые нужно сгенерировать
        max_length (int): максимальная длина среза

    Возвращает:
        List[Tuple[int, int]]: список кортежей, представляющих начальный и конечный индексы каждого среза
    """
    if start_idx == -1 and end_idx == -1:
        return []
    slice_length = end_idx - start_idx + 1

    if slice_length > max_length:
        print("Slice length cannot be greater than max_length.")
        return []

    pairs = []
    for _ in range(n):
        min_start = max(0, end_idx - max_length + 1)
        max_start = start_idx
        start = random.randint(min_start, max_start)

        min_end = end_idx
        max_end = max(start + max_length - 1, start + slice_length - 1)
        end = random.randint(min_end, max_end)

        pairs.append((start, end))

    return pairs


def generate_negative_slices(start_idx: int, end_idx: int, length: int, n: int, min_length: int, max_length: int) -> list[tuple[int, int]]:
    """
    Возвращает n срезов не включающих в себя отрезок [start_idx, end_idx] с некоторой случайностью.
    Каждый срез находится в отрезке [0, length]
    Длина каждого среза лежит в отрезке [min_length, max_length].
    Если start_idx и end_idx равны -1, то функция генерирует срезы в случайных позициях.

    Аргументы:
        start_idx (int): начальный индекс для срезов, если задан
        end_idx (int): конечный индекс для срезов, если задан
        length (int): общая длина последовательности
        n (int): количество срезов, которые нужно сгенерировать
        min_length (int): минимальная длина среза
        max_length (int): максимальная длина среза

    Возвращает:
        List[Tuple[int, int]]: список кортежей, представляющих начальный и конечный индексы каждого среза
    """

    pairs = []
    if start_idx == -1 and end_idx == -1:
        for _ in range(n):
            start = random.randint(0, max(length - max_length, 1))
            end = start + random.randint(min_length, max_length)
            pairs.append((start, end))
        return pairs
    
    assert 0 <= start_idx <= end_idx <= length
    
    if start_idx < min_length + 1 and length - end_idx < min_length + 1:
        return []

    while len(pairs) != n:
        if random.random() < 0.5 and start_idx > min_length:
            min_start = 0
            max_start = start_idx - min_length - 1
            start = random.randint(min_start, max_start)

            min_end = start + min_length
            max_end = min(start + max_length - 1, start_idx - 1)
            end = random.randint(min_end, max_end)
        elif min_length < length - end_idx - 1:
            min_start = end_idx
            max_start = length - 1 - min_length
            start = random.randint(min_start, max_start)

            min_end = start + min_length
            max_end = min(length - 1, start + max_length)
            end = random.randint(min_end, max_end)
        else:
            continue
        assert max_length > end - start >= min_length
        pairs.append((start, end))

    return pairs


def find_value_in_list(lst: list, value) -> int:
    """
    Найти индекс первого вхождения заданного значения в списке.

    Аргументы:
        lst (list): Список, в котором производится поиск.
        value: Значение, которое необходимо найти.

    Возвращает:
        int: Индекс первого вхождения значения или -1, если значение не найдено.
    """
    try:
        index = lst.index(value)
    except ValueError:
        index = -1
    return index


def pad_to_length(tensor: torch.Tensor, length: int, value):
    """
    Выполняет дополнение тензора PyTorch указанным значением до заданной длины.

    Аргументы:
        tensor (torch.Tensor): Тензор, который нужно дополнить.
        length (int): Желаемая длина дополненного тензора.
        value (float): Значение, используемое для дополнения.

    Возвращает:
        torch.Tensor: Дополненный тензор.

    Пример:
        >>> tensor = torch.tensor([1, 2, 3])
        >>> pad_to_length(tensor, 5, 0)
        tensor([1, 2, 3, 0, 0])
    """
    pad_length = length - tensor.shape[0]
    if pad_length <= 0:
        return tensor
    padded_tensor = torch.nn.functional.pad(tensor, (0, pad_length), mode='constant', value=value)
    return padded_tensor
