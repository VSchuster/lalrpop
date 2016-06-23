//! A compiler from an LR(1) table to a traditional table driven parser.

#![allow(dead_code)]

use collections::Set;
use grammar::repr::{Grammar, NonterminalString, TypeParameter, Types};
use lr1::core::*;
use lr1::lookahead::Token;
use lr1::tls::Lr1Tls;
use rust::RustWrite;
use std::io::{self, Write};
use util::{Escape, Sep};

pub fn compile<'grammar, W: Write>(grammar: &'grammar Grammar,
                                   user_start_symbol: NonterminalString,
                                   start_symbol: NonterminalString,
                                   states: &[LR1State<'grammar>],
                                   out: &mut RustWrite<W>)
                                   -> io::Result<()> {
    let _lr1_tls = Lr1Tls::install(grammar.terminals.clone());
    let mut table_driven = TableDriven::new(grammar,
                                            user_start_symbol,
                                            start_symbol,
                                            states,
                                            out);
    table_driven.write()
}

// For each state, we will create a table. The table is indexed by the
// index of the next token. The value in the table is an `i32`, and
// its interpretation varies depending on whether it is positive
// or negative:
//
// - if zero, parse error.
// - if a positive integer (not zero), it is the next state to shift to.
// - if a negative integer (not zero), it is the index of a reduction
//   action to execute (actually index + 1).
//
// We maintain two stacks: one is a stack of state indexes (each an
// u32). The other is a stack of values and spans: `(L, T, L)`. `L` is
// the location type and represents the start/end span. `T` is the
// value of the symbol. The type `T` is an `enum` that we synthesize
// which contains a variant for all the possibilities:
//
// ```
// enum Value<> {
//     // One variant for each terminal:
//     Term1(Ty1),
//     ...
//     TermN(TyN),
//
//     // One variant for each nonterminal:
//     Nt1(Ty1),
//     ...
//     NtN(TyN),
// }
// ```
//
// The table is a two-dimensional matrix indexed first by state
// and then by the terminal index. The value is

struct TableDriven<'ascent, 'grammar: 'ascent, W: Write + 'ascent> {
    /// the complete grammar
    grammar: &'grammar Grammar,

    /// some suitable prefix to separate our identifiers from the user's
    prefix: &'grammar str,

    /// types from the grammar
    types: &'grammar Types,

    /// the start symbol S the user specified
    user_start_symbol: NonterminalString,

    /// the synthetic start symbol S' that we specified
    start_symbol: NonterminalString,

    /// the vector of states
    states: &'ascent [LR1State<'grammar>],

    /// where we write output
    out: &'ascent mut RustWrite<W>,

    /// type parameters for the `Nonterminal` type
    symbol_type_params: Vec<TypeParameter>,
}

impl<'ascent, 'grammar, W: Write> TableDriven<'ascent, 'grammar, W> {
    fn new(grammar: &'grammar Grammar,
           user_start_symbol: NonterminalString,
           start_symbol: NonterminalString,
           states: &'ascent [LR1State<'grammar>],
           out: &'ascent mut RustWrite<W>)
           -> Self {
        // The nonterminal type needs to be parameterized by all the
        // type parameters that actually appear in the types of
        // nonterminals.  We can't just use *all* type parameters
        // because that would leave unused lifetime/type parameters in
        // some cases.
        let referenced_ty_params: Set<TypeParameter> = grammar.types
            .nonterminal_types()
            .into_iter()
            .chain(grammar.types.terminal_types())
            .flat_map(|t| t.referenced())
            .collect();

        let symbol_type_params: Vec<_> = grammar.type_parameters
            .iter()
            .filter(|t| referenced_ty_params.contains(t))
            .cloned()
            .collect();

        TableDriven {
            grammar: grammar,
            prefix: &grammar.prefix,
            types: &grammar.types,
            states: states,
            user_start_symbol: user_start_symbol,
            start_symbol: start_symbol,
            out: out,
            symbol_type_params: symbol_type_params,
        }
    }

    fn write(&mut self) -> io::Result<()> {
        rust!(self.out, "");
        rust!(self.out, "mod {}parse{} {{", self.prefix, self.start_symbol);

        // these stylistic lints are annoying for the generated code,
        // which doesn't follow conventions:
        rust!(self.out,
              "#![allow(non_snake_case, non_camel_case_types, unused_mut, unused_variables, \
               unused_imports)]");
        rust!(self.out, "");

        try!(self.write_uses());

        try!(self.write_value_type_defn());

        try!(self.write_parse_table());

        try!(self.write_parser_fn());

        rust!(self.out, "}}");
        Ok(())
    }

    fn write_uses(&mut self) -> io::Result<()> {
        try!(self.out.write_uses("super::", &self.grammar));

        if self.grammar.intern_token.is_none() {
            rust!(self.out, "use super::{}ToTriple;", self.prefix);
        }

        Ok(())
    }

    fn write_value_type_defn(&mut self) -> io::Result<()> {
        // sometimes some of the variants are not used, particularly
        // if we are generating multiple parsers from the same file:
        rust!(self.out, "#[allow(dead_code)]");
        rust!(self.out,
              "pub enum {}Symbols<{}> {{",
              self.prefix,
              Sep(", ", &self.symbol_type_params));

        // make one variant per terminal
        for &term in &self.grammar.terminals.all {
            let ty = self.types.terminal_type(term).clone();
            rust!(self.out, "Term{}({}),", Escape(term), ty);
        }

        // make one variant per nonterminal
        for &nt in self.grammar.nonterminals.keys() {
            let ty = self.types.nonterminal_type(nt).clone();
            rust!(self.out, "Nt{}({}),", Escape(nt), ty);
        }

        rust!(self.out, "}}");
        Ok(())
    }

    fn write_parse_table(&mut self) -> io::Result<()> {
        // The table is a two-dimensional matrix indexed first by state
        // and then by the terminal index. The value is
        rust!(self.out, "const {}ACTION: &'static [i32] = &[", self.prefix);

        for state in self.states {
            // Write an action for each terminal (either shift, reduce, or error).
            for &terminal in &self.grammar.terminals.all {
                if let Some(new_state) = state.shifts.get(&terminal) {
                    rust!(self.out, "{},", new_state.0 + 1);
                } else {
                    try!(self.write_reduction(state, Token::Terminal(terminal)));
                }
            }

            // Finally the action for EOF, which can only be a reduction.
            try!(self.write_reduction(state, Token::EOF));
        }

        rust!(self.out, "];");

        // The goto table is indexed by state and *nonterminal*.
        rust!(self.out, "const {}GOTO: &'static [i32] = &[", self.prefix);
        for state in self.states {
            for nonterminal in self.grammar.nonterminals.keys() {
                if let Some(&new_state) = state.gotos.get(nonterminal) {
                    rust!(self.out, "{},", new_state.0 + 1);
                } else {
                    rust!(self.out, "0,");
                }
            }
        }
        rust!(self.out, "];");

        Ok(())
    }

    fn write_reduction(&mut self, state: &LR1State, token: Token) -> io::Result<()> {
        let reduction = state.reductions
            .iter()
            .filter(|&&(ref t, _)| t.contains(token))
            .map(|&(_, p)| p)
            .next();
        if let Some(production) = reduction {
            let action = production.action.index();
            rust!(self.out, "-{},", action + 1);
        } else {
            // Otherwise, this is an error. Store 0.
            rust!(self.out, "0,");
        }
        Ok(())
    }

    fn write_parser_fn(&mut self) -> io::Result<()> {
        let error_type = self.types.error_type();
        let parse_error_type = self.parse_error_type();

        let (type_parameters, parameters);

        if self.grammar.intern_token.is_some() {
            // if we are generating the tokenizer, we just need the
            // input, and that has already been added as one of the
            // user parameters
            type_parameters = vec![];
            parameters = vec![];
        } else {
            // otherwise, we need an iterator of type `TOKENS`
            let mut user_type_parameters = String::new();
            for type_parameter in &self.grammar.type_parameters {
                user_type_parameters.push_str(&format!("{}, ", type_parameter));
            }
            type_parameters = vec![
                format!("{}TOKEN: {}ToTriple<{}Error={}>",
                        self.prefix, self.prefix, user_type_parameters, error_type),
                format!("{}TOKENS: IntoIterator<Item={}TOKEN>", self.prefix, self.prefix)];
            parameters = vec![format!("{}tokens: {}TOKENS", self.prefix, self.prefix)];
        }

        try!(self.out.write_pub_fn_header(
            self.grammar,
            format!("parse_{}", self.user_start_symbol),
            type_parameters,
            parameters,
            format!("Result<{}, {}>",
                    self.types.nonterminal_type(self.start_symbol),
                    parse_error_type),
            vec![]));
        rust!(self.out, "{{");

        if self.grammar.intern_token.is_some() {
            // if we are generating the tokenizer, create a matcher as our input iterator
            rust!(self.out, "let mut {}tokens = super::{}intern_token::{}Matcher::new(input);",
                  self.prefix, self.prefix, self.prefix);
        } else {
            // otherwise, convert one from the `IntoIterator`
            // supplied, using the `ToTriple` trait which inserts
            // errors/locations etc if none are given
            rust!(self.out, "let {}tokens = {}tokens.into_iter();", self.prefix, self.prefix);
            rust!(self.out, "let mut {}tokens = {}tokens.map(|t| {}ToTriple::to_triple(t));",
                  self.prefix, self.prefix, self.prefix);
        }

        rust!(self.out, "let mut {}lookahead;", self.prefix);
        rust!(self.out, "let mut {}states = vec![0_i32]", self.prefix);
        rust!(self.out, "let mut {}state_data = vec![]", self.prefix);
        try!(self.next_token("lookahead", "tokens"));
        rust!(self.out, "while let Some({}state) = states.last() {{", self.prefix);

        // first, determine which kind of token and extract an integer
        try!(self.token_to_i32("integer", "lookahead"));
        rust!(self.out, "let {}action = {}ACTION[{}state * {} + {}integer];",
              self.prefix, self.prefix,
              state,
              self.grammar.terminals.all.len() + 1,
              integer);

        // Shift.
        rust!(self.out, "if {}action > 0 {{", self.prefix);
        rust!(self.out, "{}states.push({}action - 1);", self.prefix, self.prefix);
        rust!(self.out, "{}states.push({}action - 1);", self.prefix, self.prefix);

        // Reduce.
        rust!(self.out, "}} else if {}action < 0 {{");

        // Error.
        rust!(self.out, "}} else {{");

        rust!(self.out, "}}"); // if-else-if-else

        rust!(self.out, "}}"); // while let Some(_) = ...


        rust!(self.out, "}}");
    }

    fn next_token(&mut self, lookahead: &str, tokens: &str) -> io::Result<()> {
        rust!(self.out, "{}{} = match {}{}.next() {{",
              self.prefix, lookahead, self.prefix, tokens);
        rust!(self.out, "Some(Ok(v)) => Some(v),");
        rust!(self.out, "None => None,");
        if self.grammar.intern_token.is_some() {
            // when we generate the tokenizer, the generated errors are `ParseError` values
            rust!(self.out, "Some(Err(e)) => return Err(e),");
        } else {
            // otherwise, they are user errors
            rust!(self.out, "Some(Err(e)) => return Err({}ParseError::User {{ error: e }}),",
                  self.prefix);
        }
        rust!(self.out, "}};");
        Ok(())
    }

    fn token_to_i32(&mut self, integer: &str, lookahead: &str) -> io::Result<()> {
        rust!(self.out, "let {}{} = match {}{} {{",
              self.prefix, integer,
              self.prefix, lookahead);
        for (&terminal, index) in self.grammar.terminals.all.iter().zip(0..) {
            let mut pattern_names = vec![];
            let pattern = self.match_terminal_pattern(terminal);
            rust!(self.out, "Some({}) => {},", pattern, index);
        }
        rust!(self.out, "None => {}", self.grammar.terminals.all.len());
        rust!(self.out, "}};");
    }

    /// Emit a pattern that matches `id` but doesn't extract any data.
    fn match_terminal_pattern(&mut self, id: TerminalString) -> String {
        let pattern = self.grammar.pattern(id)
                                  .map(&mut |_| "_");
        let pattern = format!("{}", pattern);
        format!("(_, {}, _)", pattern)
    }

    /// Emit a pattern that matches the terminal 
    fn match_terminal_pattern(&mut self, id: TerminalString) -> String {
        let pattern = self.grammar.pattern(id)
                                  .map(&mut |_| "_");
        let pattern = format!("{}", pattern);
        format!("(_, {}, _)", pattern)
    }


}
