# Write a module-definition file that exports the explicit instantiations of function templates
# from a static archive: the public symbols whose MSVC-decorated names begin with ??$<SYMBOL>@, where
# SYMBOL is a regular expression for the templates' names.
cmake_minimum_required(VERSION 3.28)
if(NOT DUMPBIN OR NOT EXISTS "${ARCHIVE}" OR NOT SYMBOL OR NOT EXPECTED OR NOT OUTPUT)
  message(FATAL_ERROR "Archive exports require dumpbin, an archive, a symbol, a count and an output")
endif()
execute_process(COMMAND "${DUMPBIN}" /nologo /linkermember:1 "${ARCHIVE}"
  OUTPUT_VARIABLE listing RESULT_VARIABLE result)
if(NOT result STREQUAL "0")
  message(FATAL_ERROR "dumpbin failed for ${ARCHIVE}: ${result}")
endif()
# dumpbin prints one public symbol per line after its offset. A lambda or local entity of an
# instantiation carries the instantiation's decorated name inside its own, so only a name that starts
# the symbol column counts.
string(REGEX MATCHALL "[ \t]\\?\\?\\$${SYMBOL}@[^\r\n\t ]+" names "${listing}")
list(TRANSFORM names STRIP)
list(REMOVE_DUPLICATES names)
list(LENGTH names count)
if(NOT count EQUAL EXPECTED)
  message(FATAL_ERROR "${ARCHIVE} defines ${count} instantiations of ${SYMBOL}, expected ${EXPECTED}")
endif()
list(JOIN names "\n" body)
file(WRITE "${OUTPUT}" "EXPORTS\n${body}\n")
message(STATUS "Exporting ${count} instantiations of ${SYMBOL} from ${ARCHIVE}")
