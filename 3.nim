import math
import random
import sequtils
import std/algorithm

# Compile the required C files
#{.compile: "cec2017/src/cec2017.c".}
{.compile: "cec/cec2017nim.c".}
{.compile: "cec/src/affine_trans.c".}
{.compile: "cec/src/basic_funcs.c".}
{.compile: "cec/src/cec.c".}
{.compile: "cec/src/complex_funcs.c".}
{.compile: "cec/src/hybrid_funcs.c".}
{.compile: "cec/src/interfaces.c".}
{.compile: "cec/src/utils.c".}

proc cec2017(nx: cint, fn: cint, input: ptr cdouble): cdouble {.importc: "cec2017".}

type
  Individual = seq[float]
  Population = seq[Individual]
  History = seq[Population]

const
  F = sqrt(1.0 / 2.0)
  c = 0.1
  delta = 0.01
  epsilon = 0.001

proc objectiveFunction(individual: Individual, fn_i: int): float =
  result = 0.0
  let fn = cint(fn_i)
  let nx = cint(individual.len)
  var input: seq[cdouble] = newSeq[cdouble](individual.len)
  for i in 0..<individual.len:
    input[i] = individual[i]
  result = cec2017(nx, fn, addr(input[0]))

proc initializePopulation(populationSize, dimension: int): Population =
  result = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    result[i] = newSeqWith(dimension, rand(-100.0, 100.0))

proc calculateCenter(population: Population, size: int): Individual =
  let dimension = population[0].len
  result = newSeqWith(dimension, 0.0)
  for i in 0..<size:
    for j in 0..<dimension:
      result[j] += population[i][j]
  for j in 0..<dimension:
    result[j] /= float(size)

proc sortPopulationByFitness(population: var Population, fn_i: int) =
  population.sort(proc (a, b: Individual): int =
    cmp(objectiveFunction(a, fn_i), objectiveFunction(b, fn_i))
  )

proc calculateShift(previousShift, s, m: Individual): Individual =
  let dimension = s.len
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = (1.0 - c) * previousShift[j] + c * (s[j] - m[j])

proc randomNormal(mu, sigma: float): float = 
  let u1 = rand(1.0)
  let u2 = rand(1.0)
  result = sqrt(-2.0 * log(u1)) * cos(2.0 * PI * u2) * sigma + mu

proc generateNewIndividual(s, shift: Individual, dimension: int): Individual =
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = s[j] + shift[j] + epsilon * randomNormal(0.0, 1.0)

proc handleBoundaryConstraints(x: Individual, lowerBound, upperBound: Individual): Individual =
  let dimension = x.len
  result = newSeqWith(dimension, 0.0)
  for i in 0..<dimension:
    if x[i] > upperBound[i]:
      result[i] = upperBound[i]
    elif x[i] < lowerBound[i]:
      result[i] = lowerBound[i]
    else:
      result[i] = x[i]

proc penaltyFunction(x: Individual, lowerBound, upperBound: Individual, qmax: float): float =
  let dimension = x.len
  var penalty = 0.0
  for i in 0..<dimension:
    if x[i] > upperBound[i]:
      penalty += (x[i] - upperBound[i]) ** 2
    elif x[i] < lowerBound[i]:
      penalty += (lowerBound[i] - x[i]) ** 2
  result = qmax + penalty

proc selectFromHistory(history: History, populationSize: int): Population =
  result = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    let histIndex = rand(0, history.len - 1)
    let popIndex = rand(0, history[histIndex].len - 1)
    result[i] = history[histIndex][popIndex]

proc differentialEvolutionStrategy(dimension, fn_i, maxGenerations: int): Individual =
  let populationSize = 4 * dimension
  let mu = populationSize div 2
  var population = initializePopulation(populationSize, dimension)
  var shift = newSeqWith(dimension, 0.0)
  var history: History = newSeq[Population]()
  let lowerBound = newSeqWith(dimension, -100.0)
  let upperBound = newSeqWith(dimension, 100.0)
  var qmax = 1.0e8  # Initialize with a large value

  for generation in 0..<maxGenerations:
    let m = calculateCenter(population, populationSize)
    sortPopulationByFitness(population, fn_i)
    let s = calculateCenter(population, mu)
    shift = calculateShift(shift, s, m)

    var newPopulation = newSeq[Individual](populationSize)
    for i in 0..<populationSize:
      let selectedIndividuals = selectFromHistory(history, 2)
      let a = selectedIndividuals[0]
      let b = selectedIndividuals[1]
      var d = newSeqWith(dimension, 0.0)
      for j in 0..<dimension:
        d[j] = F * (a[j] - b[j]) + shift[j] * delta * randomNormal(0.0, 1.0)
      var newIndividual = generateNewIndividual(s, d, dimension)
      newIndividual = handleBoundaryConstraints(newIndividual, lowerBound, upperBound)
      newPopulation[i] = newIndividual
    population = newPopulation

    # Update history
    history.add(population)
    if history.len > (6 + 3 * sqrt(dimension).int):
      history.del(0)  # Maintain the history size as per the requirement

    # Update qmax
    for individual in population:
      let fitness = objectiveFunction(individual, fn_i)
      if fitness < qmax:
        qmax = fitness

    # Convergence condition
    var stdDev = newSeqWith(dimension, 0.0)
    for j in 0..<dimension:
      for i in 0..<populationSize:
        stdDev[j] += pow((population[i][j] - s[j]), 2)
      stdDev[j] = sqrt(stdDev[j] / float(populationSize - 1))
    if stdDev.allIt(it < epsilon):
      break

  return population[0]

proc des*(dimension, fn_i, maxGenerations: SEXP): SEXP {.exportR.} =
  let
    dimension = dimension.to(int)
    maxGenerations = maxGenerations.to(int)
    fn_i = fn_i.to(int)
  result = nimToR(differentialEvolutionStrategy(dimension, fn_i, maxGenerations))

when isMainModule:
  let dimension = 10
  let maxGenerations = 10000
  let bestIndividual = differentialEvolutionStrategy(dimension, 1, maxGenerations)
  echo "Best individual: ", bestIndividual
  echo "Best fitness: ", objectiveFunction(bestIndividual, 1)
