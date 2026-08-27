import unittest

from aurora import AuroraHybrid, Example, AuroraMemory


class AuroraTests(unittest.TestCase):
    def test_exact_replay(self):
        data = [
            Example((1, 2, 3, 0), 17, "fact"),
            Example((4, 5, 6, 0), 23, "fact"),
            Example((1, 2, 3, 0), 17, "fact"),
        ]
        model = AuroraHybrid(64, n_key_a=17, n_key_b=19)
        model.fit(data)
        self.assertEqual(model.predict((1, 2, 3, 0))[0], 17)
        self.assertEqual(model.predict((4, 5, 6, 0))[0], 23)
        self.assertTrue(model.predict((1, 2, 3, 0))[2])

    def test_unknown_is_an_honest_miss(self):
        model = AuroraHybrid(32, n_key_a=7, n_key_b=11)
        model.fit([Example((1, 1, 1, 1), 9, "fact")])
        prediction, ops, hit = model.predict((2, 2, 2, 2))
        self.assertFalse(hit)
        self.assertEqual(ops, 1)
        self.assertEqual(prediction, 9)

    def test_hash_address_collision_does_not_make_a_false_hit(self):
        memory = AuroraMemory(
            vocab_size=32,
            n_key_a=1,
            n_key_b=1,
            max_buckets=3,
        )
        memory.update((1, 2, 3), 4)
        memory.update((5, 6, 7), 8)
        prior = [0] * 32
        prior[3] = 100
        self.assertEqual(memory.predict((1, 2, 3), prior)[0], 4)
        # Same product-key address, different fingerprint: miss, not a false
        # retrieval of token 4 or 8.
        prediction, _, hit = memory.predict((99, 99, 99), prior)
        self.assertFalse(hit)
        self.assertEqual(prediction, 3)

    def test_bounded_overflow_is_retrievable(self):
        memory = AuroraMemory(
            vocab_size=32,
            n_key_a=1,
            n_key_b=2,
            max_buckets=1,
        )
        first = (1, 2, 3)
        # Find a second context with the same primary address.  With one
        # primary bucket dimension and two secondary buckets, its deterministic
        # overflow address is different and therefore testable.
        second = None
        for i in range(2, 100):
            candidate = (i, i + 1, i + 2)
            if memory.address(first) == memory.address(candidate):
                second = candidate
                break
        self.assertIsNotNone(second)
        memory.update(first, 4)
        memory.update(second, 8)
        prior = [0] * 32
        self.assertEqual(memory.predict(second, prior)[0], 8)

    def test_explicit_online_rewrite_is_local(self):
        model = AuroraHybrid(32, n_key_a=17, n_key_b=19)
        model.fit([
            Example((1, 2, 3, 0), 4, "fact"),
            Example((9, 9, 9, 0), 10, "fact"),
        ])
        self.assertTrue(model.memory.rewrite((1, 2, 3, 0), 7))
        self.assertEqual(model.predict((1, 2, 3, 0))[0], 7)
        self.assertEqual(model.predict((9, 9, 9, 0))[0], 10)


if __name__ == "__main__":
    unittest.main()
