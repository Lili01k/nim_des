# Importy
import rnim/src/rnim
import math
import random
import sequtils
import std/algorithm

# Kompilacja do biblioteki współdzielonej
{.compile: "cec/cec2017nim.c".}
{.compile: "cec/src/affine_trans.c".}
{.compile: "cec/src/basic_funcs.c".}
{.compile: "cec/src/cec.c".}
{.compile: "cec/src/complex_funcs.c".}
{.compile: "cec/src/hybrid_funcs.c".}
{.compile: "cec/src/interfaces.c".}
{.compile: "cec/src/utils.c".}

# Deklaracja funkcji importowanej z C
proc cec2017(nx: cint, fn: cint, input: ptr cdouble): cdouble {.importc: "cec2017".}

# Typy danych
type
  Individual = seq[float]
  Population = seq[Individual]
  History = seq[Population]

# Stałe
const
  F = sqrt(1.0 / 2.0)
  c = 0.5
  delta = 1.25
  epsilon = pow(10.0,-8.0)/delta

# Zmienne globalne
var qmax = -Inf  
var bestValue = Inf
var bestIndividual: seq[float] = @[]

# Funkcja obliczeniowa (dostosowana do Twoich potrzeb)
proc objectiveFunction(individual: Individual, fn_i: int, lower, upper: Individual): float =
  result = 0.0
  if (individual.allIt(it < 100)) and (individual.allIt(it > -100)):
    let fn = cint(fn_i)
    let nx = cint(individual.len)
    var input: seq[cdouble] = newSeq[cdouble](individual.len)
    for i in 0..<individual.len:
      input[i] = individual[i]
    result = cec2017(nx, fn, addr(input[0]))
    qmax = max(qmax, result)
    if result < bestValue:
      bestValue = result
      bestIndividual = individual
  else:
    var sumSquares = 0.0
    for x in individual:
      if x > 100:
        sumSquares += (x - 100) ^ 2
      if x < -100:
        sumSquares += (-100 - x) ^ 2
    result = qmax + sumSquares

# Inicjalizacja populacji
proc initializePopulation(populationSize, dimension: int): Population =
  result = newSeq[Individual](populationSize)
  for i in 0..<populationSize:
    result[i] = newSeqWith(dimension, rand(-100.0..100.0))

# Obliczenie środka populacji
proc calculateCenter(population: Population, size: int): Individual =
  let dimension = population[0].len
  result = newSeqWith(dimension, 0.0)
  for i in 0..<size:
    for j in 0..<dimension:
      result[j] += population[i][j]
  for j in 0..<dimension:
    result[j] /= float(size)

# Sortowanie populacji względem fitnessu
proc sortPopulationByFitness(population: var Population, fn_i: int, lower, upper: Individual) =
  population.sort(proc (a, b: Individual): int =
    cmp(objectiveFunction(a, fn_i, lower, upper), objectiveFunction(b, fn_i, lower, upper)))

# Obliczenie przesunięcia
proc calculateShift(previousShift, s, m: Individual): Individual =
  let dimension = s.len
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = (1.0 - c) * previousShift[j] + c * (s[j] - m[j])

# Generowanie nowego osobnika
proc generateNewIndividual(s, shift: Individual, dimension: int): Individual =
  result = newSeqWith(dimension, 0.0)
  for j in 0..<dimension:
    result[j] = s[j] + shift[j] + epsilon * gauss(0.0, 1.0)

# Wybór z historii
proc selectFromHistory(history: History, size, mu: int): Population =
  result = newSeq[Individual](size)
  for i in 0..<size:
    let histIndex = rand(history.len - 1)
    let popIndex = rand(mu)
    result[i] = history[histIndex][popIndex]

# Strategia ewolucji różnicowej
proc differentialEvolutionStrategy(dimension, fn_i, maxGenerations: int): Individual =
  let populationSize = 4 * dimension
  let mu = populationSize div 2
  var population = initializePopulation(populationSize, dimension)
  var shift = newSeqWith(dimension, 0.0)
  var history: History = newSeq[Population]()
  let lowerBound = newSeqWith(dimension, -100.0)
  let upperBound = newSeqWith(dimension, 100.0)
  var previousMean = newSeqWith(dimension, 0.0)

  history.add(population)
  for generation in 0..<maxGenerations:
    qmax = -Inf
    let m = calculateCenter(population, populationSize)
    sortPopulationByFitness(population, fn_i, lowerBound, upperBound)
    let s = calculateCenter(population, mu)
    shift = calculateShift(shift, s, m)

    var newPopulation = newSeq[Individual](populationSize)
    for i in 0..<populationSize:
      let selectedIndividuals = selectFromHistory(history, 2, mu)
      let a = selectedIndividuals[0]
      let b = selectedIndividuals[1]
      var d = newSeqWith(dimension, 0.0)
      for j in 0..<dimension:
        d[j] = F * (a[j] - b[j]) + shift[j] * delta * gauss(0.0, 1.0)
      var newIndividual = generateNewIndividual(s, d, dimension)
      newPopulation[i] = newIndividual
    population = newPopulation

    # Aktualizacja historii
    history.add(population)
    if history.len > (6 + 3 * sqrt(float(dimension)).int):
      history.del(0)  # Zachowanie wymaganego rozmiaru historii

    # Warunek zbieżności
    var stdDev = newSeqWith(dimension, 0.0)
    for j in 0..<dimension:
      for i in 0..<populationSize:
        stdDev[j] += pow((population[i][j] - previousMean[j]), 2)
      stdDev[j] = sqrt(stdDev[j] / float(populationSize - 1))
    previousMean = s
    var err = sum(stdDev) * 0.5
    if err < epsilon:
      break

  return bestIndividual

# Eksport funkcji dla R
proc des*(dimension, fn_i, maxGenerations: cint): seq[float] {.exportc, dynlib.} =
  differentialEvolutionStrategy(dimension, fn_i, maxGenerations)

# Główny moduł
when isMainModule:
  let dimension = 2
  let maxGenerations = 2222
  let fn = 9
  let bestInd = differentialEvolutionStrategy(dimension, fn, maxGenerations)
  let lowerBound = newSeqWith(dimension, -100.0)
  let upperBound = newSeqWith(dimension, 100.0)
  echo "Best individual: ", bestInd
  echo "Best fitness: ", objectiveFunction(bestInd, fn, lowerBound, upperBound)
  echo bestValue
  echo bestInd
  echo qmax
