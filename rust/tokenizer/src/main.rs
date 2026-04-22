use std::io::{self, Read};

fn is_boundary(ch: char) -> bool {
    ch.is_whitespace() || ch.is_ascii_punctuation()
}

fn segment_class(ch: char) -> &'static str {
    if ch.is_ascii_digit() {
        "digit"
    } else if ch.is_alphabetic() {
        if ch.is_ascii() && ch.is_uppercase() {
            "upper"
        } else if ch.is_ascii() && ch.is_lowercase() {
            "lower"
        } else {
            "non_ascii"
        }
    } else {
        "other"
    }
}

fn chars_per_token(segment: &str) -> usize {
    if segment.chars().any(|ch| !ch.is_ascii()) {
        2
    } else if segment.chars().all(|ch| ch.is_ascii_digit()) {
        3
    } else if segment.chars().any(|ch| ch.is_ascii_digit()) && segment.chars().any(|ch| ch.is_ascii_alphabetic()) {
        3
    } else if segment.chars().any(|ch| ch.is_ascii_uppercase()) && segment.chars().any(|ch| ch.is_ascii_lowercase()) {
        4
    } else {
        5
    }
}

fn count_segment_tokens(segment: &str) -> usize {
    let chars: Vec<char> = segment.chars().collect();
    if chars.is_empty() {
        return 0;
    }

    let mut tokens = 0usize;
    let mut run: Vec<char> = vec![chars[0]];
    let mut prev_class = segment_class(chars[0]);

    for ch in chars.into_iter().skip(1) {
        let cls = segment_class(ch);
        let boundary = (prev_class == "lower" && cls == "upper")
            || ((prev_class == "digit" && (cls == "lower" || cls == "upper"))
                || (cls == "digit" && (prev_class == "lower" || prev_class == "upper")));
        if boundary {
            let run_str: String = run.iter().collect();
            let cpt = chars_per_token(&run_str);
            tokens += run.len().div_ceil(cpt);
            run.clear();
        }
        run.push(ch);
        prev_class = cls;
    }

    if !run.is_empty() {
        let run_str: String = run.iter().collect();
        let cpt = chars_per_token(&run_str);
        tokens += run.len().div_ceil(cpt);
    }
    tokens
}

fn count_tokens(text: &str) -> usize {
    let mut tokens = 0usize;
    let mut current = String::new();

    for ch in text.chars() {
        if is_boundary(ch) {
            if !current.is_empty() {
                tokens += count_segment_tokens(&current);
                current.clear();
            }
            if ch.is_ascii_punctuation() {
                tokens += 1;
            }
        } else {
            current.push(ch);
        }
    }

    if !current.is_empty() {
        tokens += count_segment_tokens(&current);
    }

    tokens.max(1)
}

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).expect("stdin");
    println!("{}", count_tokens(input.trim_end()));
}

#[cfg(test)]
mod tests {
    use super::count_tokens;

    #[test]
    fn punctuation_counts() {
        assert_eq!(count_tokens("hello, world!"), 4);
    }

    #[test]
    fn longer_words_split_into_subtokens() {
        assert_eq!(count_tokens("admissioncontroller"), 4);
    }

    #[test]
    fn camel_case_and_digits_split_more_aggressively() {
        assert_eq!(count_tokens("AdmissionPolicyV2"), 7);
    }
}
