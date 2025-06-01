CREATE TABLE Words (
  id INTEGER PRIMARY KEY,
  word TEXT UNIQUE NOT NULL,
  emb BLOB
);

INSERT INTO Words (id, word, emb) VALUES (1, 'apple', x_vec('0.1, 0.9, 0.0, 1.0, 0.5, 0.6'));
INSERT INTO Words (id, word, emb) VALUES (2, 'banana', x_vec('0.0, 0.8, 0.1, 0.9, 0.6, 0.5'));
INSERT INTO Words (id, word, emb) VALUES (3, 'carrot', x_vec('0.05, 0.7, 0.2, 0.8, 0.7, 0.4'));
INSERT INTO Words (id, word, emb) VALUES (4, 'date', x_vec('0.0, 0.1, 0.3, 0.7, 0.8, 0.3'));
INSERT INTO Words (id, word, emb) VALUES (5, 'elderberry', x_vec('0.5, 0.5, 0.4, 0.6, 0.9, 0.2'));
INSERT INTO Words (id, word, emb) VALUES (6, 'fig', x_vec('0.3, 0.4, 0.5, 0.5, 1.0, 0.1'));
INSERT INTO Words (id, word, emb) VALUES (7, 'grape', x_vec('0.5, 0.3, 0.6, 0.4, 0.9, 0.2'));
INSERT INTO Words (id, word, emb) VALUES (8, 'honeydew', x_vec('0.8, 0.2, 0.7, 0.3, 0.8, 0.3'));
INSERT INTO Words (id, word, emb) VALUES (9, 'kiwi', x_vec('0.3, 0.1, 0.8, 0.2, 0.7, 0.4'));
INSERT INTO Words (id, word, emb) VALUES (10, 'lemon', x_vec('1.0, 0.0, 0.9, 0.1, 0.6, 0.5'));
INSERT INTO Words (id, word, emb) VALUES (11, 'wild banana', x_vec('0.05, 0.7, 0.1, 0.9, 0.6, 0.5'));
