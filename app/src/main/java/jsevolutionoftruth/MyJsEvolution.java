package jsevolutionoftruth;

import static java.util.Objects.requireNonNull;
import static java.util.stream.Collectors.joining;

import java.util.List;
import java.util.Map;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.Stream;

import javax.script.ScriptEngine;

import org.openjdk.nashorn.api.scripting.NashornScriptEngineFactory;

import io.jenetics.IntegerGene;
import io.jenetics.Phenotype;
import io.jenetics.SinglePointCrossover;
import io.jenetics.SwapMutator;
import io.jenetics.engine.Codec;
import io.jenetics.engine.Engine;
import io.jenetics.engine.EvolutionResult;
import io.jenetics.engine.Limits;
import io.jenetics.engine.Problem;
import io.jenetics.util.IntRange;
import jsevolutionoftruth.utils.ScriptFunction;

import io.jenetics.ext.grammar.Bnf;
import io.jenetics.ext.grammar.Cfg;
import io.jenetics.ext.grammar.Cfg.Terminal;
import io.jenetics.ext.grammar.Mappers;
import io.jenetics.ext.grammar.SentenceGenerator;
import io.jenetics.ext.util.TreeNode;

import io.jenetics.prog.regression.Error;
import io.jenetics.prog.regression.LossFunction;
import io.jenetics.prog.regression.Sample;
import io.jenetics.prog.regression.Sampling;
import io.jenetics.prog.regression.Sampling.Result;

public class MyJsEvolution
        implements Problem<ScriptFunction, IntegerGene, Double> {
    /**
     * Lookup table for XOR
     */
    public static final List<Sample<Boolean>> SAMPLES = List.of(
            Sample.of(new Boolean[] { true, true, false }),
            Sample.of(new Boolean[] { true, false, true }),
            Sample.of(new Boolean[] { false, true, true }),
            Sample.of(new Boolean[] { false, false, false }));

    // Create the script engine of your choice.
    private static final ScriptEngine SCRIPT_ENGINE = new NashornScriptEngineFactory().getScriptEngine();

    // Define a grammar which creates a valid script for the script engine.
    private static final Cfg<String> GRAMMAR = Bnf.parse("""
            <expr>         ::= <var> | <boolean-expr> | <ternary-expr>
            <ternary-expr> ::= '(' <expr> ') ? (' <expr> ') : (' <expr> ')'
            <boolean-expr> ::= (<expr>) <relation> (<expr>)
            <unary>        ::= '!' <expr>
            <relation>     ::= ' && ' | ' || ' | ' != ' | ' == '
            <var>          ::= x | y
            """);

    private static final Codec<ScriptFunction, IntegerGene> CODEC = Mappers
            // Creating a GE mapper/codec: `Codec<ScriptFunction, IntegerGene>`
            .multiIntegerChromosomeMapper(
                    GRAMMAR,
                    // The length of the chromosome is 25 times the length of the
                    // alternatives of a given rule. Every rule gets its own chromosome.
                    // It would also be possible to define variable chromosome length
                    // with the returned integer range.
                    rule -> IntRange.of(rule.alternatives().size() * 25),
                    // The used generator defines the generated data type, which is
                    // `List<Terminal<String>>`.
                    index -> new SentenceGenerator<>(index, 50))
            // Map the type of the codec from `Codec<List<Terminal<String>, IntegerGene>`
            // to `Codec<String, IntegerGene>`
            .map(s -> s.stream().map(Terminal::value).collect(joining()))
            .map(script -> new ScriptFunction(script, SCRIPT_ENGINE));

    private static final Error<Double> ERROR = Error.of(LossFunction::mse);

    private final Sampling<Boolean> _sampling;

    public MyJsEvolution(final Sampling<Boolean> sampling) {
        _sampling = requireNonNull(sampling);
    }

    public MyJsEvolution(final List<Sample<Boolean>> samples) {
        this(Sampling.of(samples));
    }

    @Override
    public Codec<ScriptFunction, IntegerGene> codec() {
        return CODEC;
    }

    @Override
    public Function<ScriptFunction, Double> fitness() {
        return script -> {
            final Result<Boolean> result = _sampling.eval(args -> {
                final var value = (Boolean) script.apply(Map.of("x", args[0], "y", args[1]));
                return (Boolean) (value != null ? value : false);
            });
            return ERROR.apply(TreeNode.of(),
                    (Double[]) Stream.of(result.calculated()).map(e -> e ? 1.0 : 0.0).collect(Collectors.toList())
                            .toArray(new Double[0]),
                    (Double[]) Stream.of(result.expected()).map(e -> e ? 1.0 : 0.0).collect(Collectors.toList())
                            .toArray(new Double[0]));
        };
    }

    public static void main(final String[] args) {
        final var regression = new MyJsEvolution(SAMPLES);

        final Engine<IntegerGene, Double> engine = Engine.builder(regression)
                .alterers(new SwapMutator<>(), new SinglePointCrossover<>())
                .minimizing()
                .build();

        final EvolutionResult<IntegerGene, Double> result = engine.stream()
                .limit(Limits.byFitnessThreshold(0.05))
                .limit(10000)
                .collect(EvolutionResult.toBestEvolutionResult());

        final Phenotype<IntegerGene, Double> best = result.bestPhenotype();
        final ScriptFunction program = CODEC.decode(best.genotype());

        System.out.println("Generations: " + result.totalGenerations());
        System.out.println("Function:    " + program);
        System.out.println("Error:       " + regression.fitness().apply(program));
    }

}
