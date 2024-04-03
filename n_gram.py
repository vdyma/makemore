import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split


def load_data(
    file_path: str = "names.txt",
    test_size: int | float = 0,
    train_size: int | float = 1,
    seed: int = 0,
    special_token: str = ".",
) -> dict[str, object]:
    text = open(file_path, "r", encoding="utf8").read()
    if special_token in text:
        raise ValueError(f"Special token '{special_token}' is present in the data.")
    words = text.split("\n")
    alphabet = [special_token] + sorted(list(set("".join(words))))
    itos = dict((i, s) for i, s in enumerate(alphabet))  # index to string
    stoi = dict((s, i) for i, s in itos.items())

    train, test = (
        train_test_split(
            words, test_size=test_size, train_size=train_size, random_state=seed
        )
        if test_size != 0
        else [words, []]
    )

    return {
        "train": train,
        "test": test,
        "itos": itos,
        "stoi": stoi,
    }


def create_count_matrix(
    words: list[str],
    stoi: dict[str, int],
    special_token: str = ".",
    num_grams: int = 2,
    smoothing_factor: int = 0,
) -> torch.Tensor:
    max_word_len = max(len(word) for word in words)
    assert (
        2 <= num_grams <= max_word_len
    ), f"num_grams must be at least 2 or at most {max_word_len} for this dataset"

    shape = tuple(len(stoi) for _ in range(num_grams))
    counts = torch.zeros(shape, dtype=torch.int32) + smoothing_factor

    for word in words:
        processed_word = (
            list(special_token for _ in range(num_grams - 1))
            + list(word)
            + [special_token]
        )
        zipped_word = zip(*[processed_word[i:] for i in range(num_grams)])
        for chars in zipped_word:
            counts[tuple(map(lambda x: stoi[x], chars))] += 1

    return counts


def sample_with_counts(
    num_samples: int,
    counts: torch.Tensor,
    itos: dict[int, str],
    seed: int = 0,
) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    samples = []

    for i in range(num_samples):
        indexes = [0 for _ in range(len(counts.shape) - 1)]
        chars = []
        while True:
            ix = torch.multinomial(
                counts[tuple(indexes)], 1, replacement=True, generator=g
            ).item()
            if ix == 0:
                break

            chars.append(itos[ix])
            indexes = indexes[1:]
            indexes.append(ix)

        samples.append("".join(chars))

    return samples


def calculate_counts_loss(
    words: list[str],
    counts: torch.Tensor,
    stoi: dict[str, int],
    special_token: str = ".",
    epsilon: float = 1e-10,
) -> float:
    log_likelihood = 0.0
    n = 0

    probs = counts.float()
    probs /= probs.sum(dim=len(probs.shape) - 1, keepdim=True) + epsilon
    for word in words:
        processed_word = [special_token] + list(word) + [special_token]
        zipped_word = zip(*[processed_word[i:] for i in range(len(counts.shape))])
        for chars in zipped_word:
            log_likelihood += torch.log(
                probs[tuple(map(lambda x: stoi[x], chars))] + epsilon
            ).item()
            n += 1

    return -log_likelihood / n


def prepare_for_nn(
    words: list[str],
    stoi: dict[str, int],
    num_grams: int,
    special_token: str = ".",
) -> dict[str, torch.Tensor]:
    xs = []
    ys = []
    num_classes = len(stoi)

    for word in words:
        processed_word = [special_token] + list(word) + [special_token]
        zipped_word = zip(*[processed_word[i:] for i in range(num_grams)])
        for chars in zipped_word:
            xs.append(
                list(
                    map(
                        lambda x: stoi[x[1]] + num_classes * x[0], enumerate(chars[:-1])
                    )
                )
            )
            ys.append(stoi[chars[-1]])

    return {
        "X": torch.tensor(xs),
        "y": torch.tensor(ys),
    }


def init_nn_model(
    alphabet_length: int, num_grams: int, seed: int = 0
) -> dict[str, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    weights = torch.randn(
        (alphabet_length * (num_grams - 1), alphabet_length),
        generator=g,
        requires_grad=True,
    )
    bias = torch.randn((1, alphabet_length), generator=g, requires_grad=True)
    return {"W": weights, "b": bias}


def train_nn(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: dict[str, torch.Tensor],
    epochs: int = 100,
    alpha: float = 50.0,
    regularization_param: float = 0.01,
    verbose: bool = True,
) -> torch.Tensor:
    loss_history = torch.empty((epochs,))
    for i in range(epochs):
        logits = model["W"][inputs].sum(dim=1) + model["b"]
        loss = (
            F.cross_entropy(logits, labels)
            + regularization_param * (model["W"] ** 2).mean()
        )
        loss_history[i] = loss.item()
        if verbose:
            print(f"Epoch: {i}\tLoss: {loss.item()}")

        model["W"].grad = None
        model["b"].grad = None
        loss.backward()
        model["W"].data += -alpha * model["W"].grad
        model["b"].data += -alpha * model["b"].grad

    return loss_history


def sample_with_nn(
    num_samples: int,
    model: dict[str, torch.Tensor],
    itos: dict[int, str],
    seed: int = 0,
) -> list[str]:
    g = torch.Generator().manual_seed(seed)
    samples = []

    for i in range(num_samples):
        indexes = [0 for _ in range(model["W"].shape[0] // len(itos))]
        chars = []

        while True:
            logits = model["W"][indexes].sum(dim=1) + model["b"]
            # softmax
            counts = logits.exp()
            probs = counts / counts.sum(dim=1, keepdim=True)
            ix = torch.multinomial(probs, 1, replacement=True, generator=g).item()
            if ix == 0:
                break

            chars.append(itos[ix])
            indexes = indexes[1:]
            indexes.append(ix)

        samples.append("".join(chars))

    return samples


def calculate_nn_loss(
    inputs: torch.Tensor,
    labels: torch.Tensor,
    model: dict[str, torch.Tensor],
) -> torch.Tensor:
    logits = model["W"][inputs].sum(dim=1) + model["b"]
    return F.cross_entropy(logits, labels)
